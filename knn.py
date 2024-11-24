import numpy as np
from collections import Counter

class labeled_point:
    vec = np.array([])
    label = ''

    def __init__(self, _vec, _label):
        self.vec = _vec
        self.label = _label

class knn_classifier:
    def most_common(self, neighbors):
        counts = Counter(neighbors)
        winner, winner_count = counts.most_common(1)[0]
        num_winners = len([ i for i in counts.values() if i == winner_count ])

        if num_winners == 1:
            return winner
        else:
            return self.most_common(neighbors[:-1])
    
    def magnitude(self, vec):
        mag = np.sqrt(sum( (v**2) for v in vec ))
    
        return mag

    def distance(self, vec1, vec2):
        dis = vec1 - vec2
        return self.magnitude(dis)
    
    def knn_classify(self, k, dataSet, point):

        points = dataSet.iloc[:,:-1].to_numpy()
        labels = dataSet.iloc[:, -1]
    
        labeled_points = [labeled_point(points[i], labels[i]) for i in range(len(points))]

        by_distance = sorted(labeled_points, key= lambda lp: self.distance(point, lp.vec), reverse=False)
        neighbors = by_distance[:k]

        return self.most_common(neighbors=[ n.label for n in neighbors])
