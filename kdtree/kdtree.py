import time
import numpy as np
import copy
import math

# we generate random numbers; setting a "seed"
# will lead to the same "random" set when 
# exexuting the cell mulitple times
np.random.seed(42)

class Node():
    """ Helper class that reprents
    a single node of a k-d tree.
    """
    
    def __init__(self, left, right, points=None, indices=None,bounds=None):

        self.left = left
        self.right = right
        self.points = points
        self.indices = indices
        self.bounds = bounds


class KDTree():
    
    def __init__(self, X, leaf_size=30):
        
        self.leaf_size = leaf_size
        self._fit(X)    
    
    def _fit(self, X):
        
        self._dim = len(X[0])
        
        I = np.array(range(len(X)))
        points = copy.deepcopy(X)
        self._root = self._build_tree(points, I)

    def _build_tree(self, pts, indices, depth=0,bounds=None):
        if bounds is None:
            bounds = np.array([[-np.inf,np.inf]]*self._dim)

        if len(pts) <= self.leaf_size: 
            #TODO add storage on disk with numpy mmap 
            return Node(left=None, right=None, points=pts, indices=indices,bounds=bounds)
        
        axis = depth % self._dim
        
        part = pts[:,axis].argsort()
        indices = indices[part]
        pts = pts[part]

        midx = math.floor(len(pts)/2)
        median = pts[midx, axis]

        l_bounds,r_bounds = bounds.copy(),bounds.copy()
        l_bounds[axis,1] = median
        r_bounds[axis,0] = median

        lefttree = self._build_tree(pts[:midx,:], indices[:midx], depth+1,l_bounds)
        righttree = self._build_tree(pts[midx:,:], indices[midx:], depth+1,r_bounds)

        return Node(left=lefttree, right=righttree,bounds=bounds)

    def query_box(self,mins,maxs):
        if self._root is None:  
            raise Exception("Tree not fitted yet!")

        points = self._recursive_search(self._root,mins,maxs)

        print(points)

    def _recursive_search(self,node,mins,maxs,depth=0,points=[]):
        if (node.left == None and node.right==None):
            # is partition fully contained by box
            if (np.all(node.bounds[:,0] >= mins)) and (np.all(node.bounds[:,1] <= maxs)):
                points.append(node.indices)
                return points
            else:
                mask = (np.all(node.points >= mins,axis=1) ) &  (np.all(node.points <= maxs, axis=1))
                points.append(node.indices[mask])
                return points

        l_bounds = node.left.bounds 
        r_bounds = node.right.bounds

        #if intersects
        if not ( np.any(l_bounds[:,0] > maxs) ) or ( np.any(l_bounds[:,1] < mins )):
            self._recursive_search(node.left,mins,maxs,points)

        if not ( np.any(r_bounds[:,0] > maxs) ) or ( np.any(r_bounds[:,1] < mins )):
            self._recursive_search(node.right,mins,maxs,points)

        return points


        


        


            
        


