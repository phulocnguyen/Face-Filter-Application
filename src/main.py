import cv2
import mediapipe as mp
from scipy.spatial import Delaunay
from omegaconf import DictConfig
import hydra
import rootutils
import csv
import numpy as np
from math_utils import FBC
import math

root_path = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "config")


class FaceDetectionUtils():

    def __init__(self, face_to_index) -> None:
        self.face2index = face_to_index

    def face_detection(self, image):
        self.image = image
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7)
        result = face_detection.process(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        face_detection.close()
        return result

    def face_landmark_points(self, faces):
        landmark_points = []
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True)

        face_detection = faces
        # IS_FACE_DETECTION = len(face_detection.detections) >= 1
        IS_FACE_DETECTION = True
        if IS_FACE_DETECTION:
            for face in face_detection.detections:
                face_landmark_points = []
                face_box = face.location_data.relative_bounding_box
                ih, iw, _ = self.image.shape
                x, y, w, h = int(face_box.xmin * iw), int(face_box.ymin *
                                                          ih), int(face_box.width * iw), int(face_box.height * ih)

                face_cropped = self.image[y:y+h, x:x+w]
                landmark_points_cropped = face_mesh.process(
                    cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB))

                # IS_FACE_LANDMARK_POINTS = len(landmark_points_cropped.multi_face_landmarks) >= 1
                IS_FACE_LANDMARK_POINTS = True
                if IS_FACE_LANDMARK_POINTS:
                    for face_landmarks in landmark_points_cropped.multi_face_landmarks:
                        for points in face_landmarks.landmark:
                            x_origin = int(points.x * w) + x
                            y_origin = int(points.y * h) + y

                            face_landmark_points.append((x_origin, y_origin))
                landmark_points.append(face_landmark_points)

        face_mesh.close()
        return landmark_points

    def face_delaunay_triangulation(self, landmark):
        landmark_points = landmark
        delaunay_triangulation = []
        for face_landmark in landmark_points:
            triangle = Delaunay(face_landmark)
            delaunay_triangulation.append(triangle.simplices)
        return delaunay_triangulation

    def run(self):
        face_box = self.face_detection()
        face_landmark = self.face_landmark_points(face_box)
        delaunay = self.face_delaunay_triangulation(face_landmark)
        return {"box": face_box, "landmark": face_landmark, "delaunay": delaunay}


class FilterUtils():
    def __init__(self, config) -> None:
        self.filter_path = config.filter_path

    def __load_filter_image(self, image_directory, include_alpha: bool):
        image = cv2.imread(image_directory, cv2.IMREAD_UNCHANGED)
        alpha = None
        if include_alpha:
            b, g, r, alpha = cv2.split(image)
            image = cv2.merge((b, g, r))
        return image, alpha

    def __load_filter_landmark(self, landmark_directory: str):
        with open(landmark_directory) as file:
            csv_reader = csv.reader(file, delimiter=",")
            points = {}

            for _, rowValue in enumerate(csv_reader):
                try:
                    x, y = int(rowValue[1]), int(rowValue[2])
                    points[rowValue[0]] = (x, y)
                except ValueError:
                    continue
            return points

    def find_convex_hull(self, points):
        hull = []
        hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False,
                                   returnPoints=False)  # tính toán đường bao lồi
        addPoints = [
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [61], [146], [91], [
                181], [84], [17], [314], [405], [321], [375], [291],  # MEDIAPIPE OUTER LIPS
            [78], [95], [88], [178], [87], [14], [317], [402], [
                318], [324], [308],  # MEDIAPIPE INNNER LIPS
            [168], [197], [2], [326],  # MEDIAPIPE NOSE
            [362], [382], [381], [380], [374], [373], [390], [249], [263], [33], [
                7], [163], [144], [145], [153], [154], [155], [133],  # MEDIAPIPE EYES
            [336], [296], [334], [293], [300], [70], [63], [
                105], [66], [107]     # MEDIAPIPE EYEBROWS
        ]
        hullIndex = np.concatenate((hullIndex, addPoints))
        for i in range(0, len(hullIndex)):
            hull.append(points[str(hullIndex[i][0])])
        return hull, hullIndex

    def load_filter(self, filter_name):
        filter_data = self.filter_path[filter_name]
        filter_runtime = []
        for filter_part in filter_data:
            result_dict = {}
            filter_image, filter_alpha = self.__load_filter_image(
                filter_part['path'], filter_part['has_alpha'])
            result_dict['image'] = filter_image
            result_dict['image_alpha'] = filter_alpha

            landmark_points = self.__load_filter_landmark(
                filter_part['anno_path'])
            result_dict['landmark_point'] = landmark_points

            if (filter_part['morph']):
                hull, hullIndex = self.find_convex_hull(landmark_points)
                sizeImg1 = filter_image.shape
                rect = (0, 0, sizeImg1[1], sizeImg1[0])
                dt = FBC().calculateDelaunayTriangles(rect, hull)
                result_dict['hull'] = hull
                result_dict['hullIndex'] = hullIndex
                result_dict['delaunay'] = dt

                if len(dt) == 0:
                    continue

            if (filter_part['animated']):
                filter_animated = cv2.VideoCapture(filter_part['path'])
                result_dict['animated'] = filter_animated

            filter_runtime.append(result_dict)
        return filter_data, filter_runtime


class FaceFilterApplication():
    def __init__(
        self, config, videoPath, filter_name,
        filter_tool: FilterUtils, facial_detection_tool: FaceDetectionUtils
    ) -> None:
        self.config = config
        self.filter_name = filter_name
        self.videoPath = videoPath

        self.filter_tool = filter_tool
        self.facial_detection_tool = facial_detection_tool

    def __load_video_capture(self):
        # kiểm tra có muốn dùng chế độ thời gian thực hay ko
        if (self.config.face_filter_application.realtime):
            self.capture = cv2.VideoCapture(0)  # mở webcam
            if not self.capture.isOpened():  # nếu kh thể mở webcam
                print("Cannot open webcam")
        else:
            # nếu dùng chế độ khác thì load video
            self.capture = cv2.VideoCapture(self.videoPath)

    def __estimate_optical_flow(self, img2GrayPrevious, img2GrayCurrent, points2Previous, points2):
        lk_params = dict(winSize=(101, 101), maxLevel=15,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
        points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrevious, img2GrayCurrent, points2Previous,
                                                        np.array(
                                                            points2, np.float32),
                                                        **lk_params)
        return points2Next

    def __point2Estimate(self, points2, points2Next, frame):
        SIGMA = self.config.face_filter_application.optical_flow.sigma
        for k in range(0, len(points2)):
            # tính toán độ lệch điểm mốc ở 2 khung hình
            d = cv2.norm(np.array(points2[k]) - points2Next[k])
            alpha = math.exp(-d * d / SIGMA)
            points2[k] = (1 - alpha) * np.array(points2[k]) + \
                alpha * points2Next[k]
            points2[k] = FBC().constrainPoint(
                points2[k], frame.shape[1], frame.shape[0])
            points2[k] = (int(points2[k][0]), int(points2[k][1]))
        return points2

    def __showLandmark(self, points2, frame):
        for idx, point in enumerate(points2):
            cv2.circle(frame, point, 2, (255, 0, 0), -1)
            cv2.putText(frame, str(idx), point,
                        cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
        cv2.imshow("landmarks", frame)

    def __warpFilter(self, frame, points2, multi_filter_runtime, filters):
        for idx, filter in enumerate(filters):
            filter_runtime = multi_filter_runtime[idx]
            img1 = filter_runtime['image']
            points1 = filter_runtime['landmark_point']
            img1_alpha = filter_runtime['image_alpha']
            dst_points = [
                points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
            tform = FBC().similarityTransform(
                list(points1.values()), dst_points)[0]

            trans_img = cv2.warpAffine(
                img1, tform, (frame.shape[1], frame.shape[0]))
            trans_alpha = cv2.warpAffine(
                img1_alpha, tform, (frame.shape[1], frame.shape[0]))
            mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

            mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

            mask2 = (255.0, 255.0, 255.0) - mask1

            temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
            temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
            output = temp1 + temp2

            frame = output = np.uint8(output)
        return output

    def run(self):
        IS_FIRST_FRAME = True
        VIDEO_FACE_VISUALIZE = self.config.face_filter_application.visualize_face_points

        filters, multi_filter_runtime = self.filter_tool.load_filter(
            self.filter_name)
        self.__load_video_capture()

        while (self.capture.isOpened()):
            ret, frame = self.capture.read()
            if not ret:
                break

            face_box = self.facial_detection_tool.face_detection(frame)
            if (face_box):
                continue
            face_landmark_points = self.facial_detection_tool.face_landmark_points(face_box)[
                0]
            if not face_landmark_points:
                continue

            points2 = face_landmark_points
            img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if IS_FIRST_FRAME:
                points2Prev = np.array(points2, np.float32)
                img2GrayPrev = np.copy(img2Gray)
                IS_FIRST_FRAME = False

            points2Next = self.__estimate_optical_flow(
                img2GrayPrev, img2Gray, points2Prev, points2)
            points2 = self.__point2Estimate(points2, points2Next, frame)
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = img2Gray
            if (VIDEO_FACE_VISUALIZE):
                self.__showLandmark(points2, frame)
            output = self.__warpFilter(
                frame, points2, multi_filter_runtime, filters)
            cv2.imshow("Face Filter", output)
            keypressed = cv2.waitKey(1) & 0xFF == ord('q')
            if keypressed == 10:
                break

        self.capture.release()
        cv2.destroyAllWindows()


@hydra.main(version_base=None, config_path=config_path, config_name="filter")
def main(cfg: DictConfig) -> None:
    facial_detection_tool = FaceDetectionUtils(cfg.facial_detection_configs)
    filter_tool = FilterUtils(cfg.filter_configs)
    app = FaceFilterApplication(
        cfg, cfg.test.video_path, cfg.test.filter_name, filter_tool, facial_detection_tool)
    app.run()


if __name__ == "__main__":
    main()
