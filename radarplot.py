from typing import List, Tuple
from numpy import pi, sin, cos, tan, sqrt
from numpy import sum as Sum
from numpy import array, linspace, insert, cumsum, diff
from numpy.random import rand
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class _Basis:
    # set A to be the basis for the graph G

    def __init__(self, dim:int) -> None:

        self.dim = dim
        self.standard_basis = [(cos(2*pi*k/dim), sin(2*pi*k/dim)) for k in range(dim)]
        
        pass
    
    def __array_sizer(self, l:list) -> list:

        new_l = l
        while len(new_l) != self.dim:
            if len(new_l) > self.dim:
                new_l.pop()
            else:
                # fill with approx. zeros
                # Reason: duplicate 0 not handled well by interpolator
                new_l.append(1e-6)
        
        return new_l

    def _array_transformer(self, l:List[float]) -> List[float]:

        if (min(l) < 0) or (max(l) > 1):
            # bijection of half-plane <-> [0,1): f(x) = 1 - 1/(x+1).
            return [1-1/(abs(x)+1) for x in l]
        return l

    def angle(self, k:int) -> float:
        return 2*pi*k/self.dim

    def random_basis(self) -> list:
        
        A = []
        for k in range(self.dim):
            theta_k = self.angle(k)
            if (0 <= k <= self.dim) and ((k!=self.dim/4) and (k!=3*self.dim/4)):
                t = rand() * cos(theta_k)
                A += [(t, tan(theta_k)*t)]
                
        if self.dim % 4 == 0:
            orth_up = (0, rand())
            orth_dn = (0, -rand())
            A.insert(self.dim//4, orth_up)
            A.insert(3*self.dim//4, orth_dn)
        
        return A
    
    def normed_verticies(self, verticies:List[float]) -> List[Tuple[float,float]]:
        
        # PRECONDITION: len(verticies) == self.dim
        if len(verticies) != self.dim:
            verticies = self.__array_sizer(verticies)
        # Re-scale if elements are not [0,1]
        verticies = self._array_transformer(verticies)

        xy = []
        for k,w in enumerate(verticies):
            theta_k = self.angle(k)
            if (0 <= k <= self.dim) and ((k!=self.dim/4) and (k!=3*self.dim/4)):
                # t = w * cos(theta_k)
                t = w * self.standard_basis[k][0]
                xy += [(t, tan(theta_k) * t)]
            else:   # k == n/4, 3n/4
                if k == self.dim/4:
                    xy += [(0, w)]
                elif k == 3*self.dim/4:
                    xy += [(0, -w)]
        
        return xy
    
    pass

class Graph(_Basis):
    # set of points with basis A

    def __init__(self, dim:int, weights:List[float]) -> None:
        
        super().__init__(dim)
        # PRECONDITION: 0<= weights[i] <= 1. If not, list will be transformed.
        self.weights = super()._array_transformer(weights)
        self.basis = super().normed_verticies(self.weights)
        # Compare to area of n-gon: A = n/2 * sin(2pi/n) 
        self.area = self._area() * pi/(0.5*dim * sin(2*pi/dim))
        
        pass
    
    def __interpolation(self, method:str):

        # PRECONDITION: method must be in METHODS
        METHODS = ['slinear', 'quadratic', 'cubic']
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}.")
        
        # points to interpolate given by the verticies of the basis
        points = array([[x[0] for x in self.basis] + [self.basis[0][0]],
                        [x[1] for x in self.basis] + [0]]).T

        # Linear length along the line:
        distance = cumsum(sqrt(Sum(diff(points, axis=0)**2, axis=1)))
        distance = insert(distance, 0, 0)/distance[-1]
        
        interpolator =  interp1d(distance, points, kind=method, axis=0)
        return interpolator(linspace(0, 1, 100))

    
    def _smush(self, x:float, y:float) -> list:
        # given an tuple of shape 2. Consider P = (x,y)
        # if x^2+y^2 > 1, find the closest point on the unit disk to P
        # analytic solution
        delta = (1+(y/x)**2)**(-0.5)
        sgn = y/abs(y)
        return [delta, sgn * sqrt(1 - delta**2)]

    def _area(self) -> float:

        n = len(self.basis)
        a = 0
        
        for k in range(len(self.basis)):
            x1,y1 = self.basis[k]
            x2,y2 = self.basis[(k+1)%n]
            t1,t2 = 2*k*pi/n, 2*(k+1)*pi/n
            # vertical line segment, or a degerate point at (0,0)
            if x1 == x2:
                integral = x1**2 * (tan(t2) - tan(t1))
            # horizontal line segment
            elif y1 == y2:
                # line segment on the x-axis
                if y1 == 0:
                    integral = 0
                else:
                    integral = -y1**2 * (1/tan(t2) - 1/tan(t1))
            # any diagonal line segment
            else:
                m = (y2-y1)/(x2-x1)
                integral = (-m*x1 + y1)**2 * (1/(m-tan(t2)) - 1/(m-tan(t1)))
            
            a += integral
        return 0.5 * a

    def _patch(self, method:str, alpha:float=0.8, fill:bool=False):
        
        # Red-Yellow-Green gradient color map for plot
        cmap = plt.get_cmap('RdYlGn')
        # Colour graph based on proportion of radar plot covered
        # Scale by 1.25 to make color gradient more coarse
        color = cmap(min(0.999, 1.25 * self.area/pi))
        # Compute interpolated (x,y) points for boundary curve
        curve = self.__interpolation(method)
        # make sure the interpolation is within R
        for i in range(len(curve)):
            if curve[i][0]**2 + curve[i][1]**2 > 1:
                curve[i] = self._smush(curve[i][0], curve[i][1])

        plt.plot(*curve.T, color=color, alpha=alpha, lw=0.1)
        if fill:
            plt.fill(*curve.T, color=color, alpha=alpha/1.25)

    pass


class RadarPlot:

    def __init__(self, dim:int, labels:List[str], g:Graph) -> None:
        
        self.dim = dim
        # PRECONDITION: length(self.labels) == dim
        self.labels = labels
        while len(self.labels) != dim:
            if len(self.labels) > dim:
                self.labels.pop()
            else:
                # label verticies v_i
                self.labels.append(f"v{1+len(self.labels)}")
        
        self.basis = _Basis(dim=dim).standard_basis
        self.graph = g
        
        pass
    
    def __figure(self, spokes:bool, dpi:int):

        X, Y = list(zip(*self.basis))

        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot()
        
        # plot and annotate verticies
        plt.scatter(x=X, y=Y, s=5, color='black', alpha=0.67)
        for i, (x,y) in enumerate(self.basis):
            ax.annotate(text=f'{self.labels[i]}', xy=(1.1*x,1.1*y),
                        textcoords='offset points', ha='center', va='bottom')
        # circular outline on plot
        plt.plot(cos(linspace(0, 2*pi, 300)), sin(linspace(0, 2*pi, 300)), 'k', lw=0.25)
        
        if spokes:
            # plot spokes from the origin to the verticies
            for k in range(self.dim):
                theta_k = self.graph.angle(k)
                t = linspace(0, cos(theta_k), 200)
                plt.plot(t, tan(theta_k) * t, 'k', lw=0.5)
        
        # 1:1 aspect ratio
        ax.set_aspect('equal')
        plt.axis('off')
        
        pass

    def plot(self, interpolation:str='quadratic', spokes:bool=True, fill:bool=False, alpha:float=0.8, dpi:int=100):

        self.__figure(spokes, dpi)
        self.graph._patch(method=interpolation, fill=fill, alpha=alpha)
        plt.show()

    pass

