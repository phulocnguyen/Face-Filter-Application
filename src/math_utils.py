import math
import cv2
import numpy as np

class FBC(): 
  def __init__(self) -> None:
    pass
  
  def constrainPoint(self, p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p
  
  def rectContains(self, rect, point):
      if point[0] < rect[0]:
        return False
      elif point[1] < rect[1]:
        return False
      elif point[0] > rect[2]:
        return False
      elif point[1] > rect[3]:
        return False
      return True
    
  def calculateDelaunayTriangles(self, rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
      subdiv.insert((p[0], p[1]))
      
    triangleList = subdiv.getTriangleList()
    delaunayTri = []

    for t in triangleList:
      pt = []
      pt.append((t[0], t[1]))
      pt.append((t[2], t[3]))
      pt.append((t[4], t[5]))

      pt1 = (t[0], t[1])
      pt2 = (t[2], t[3])
      pt3 = (t[4], t[5])

      if self.rectContains(rect, pt1) and self.rectContains(rect, pt2) and self.rectContains(rect, pt3):
        # Variable to store a triangle as indices from list of points
        ind = []
        # Find the index of each vertex in the points list
        for j in range(0, len(pt)):
          for k in range(0, len(points)):
            if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
              ind.append(k)
          # Store triangulation as a list of indices
        if len(ind) == 3:
          delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri
  
  def similarityTransform(self, inPoints, outPoints):
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    # The third point is calculated so that the three points make an equilateral triangle
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([int(xin), int(yin)])

    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([int(xout), int(yout)])

    # Now we can use estimateRigidTransform for calculating the similarity transform.
    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]), False)
    return tform