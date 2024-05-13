#!/usr/bin/env python3

"""Simple Graph implementation."""

from typing import Union
from collections import defaultdict
import sys
import numpy as np
sys.dont_write_bytecode=True

__all__=["Graph"]

class Graph:
    """Graph class implementation."""
    def __init__(self,nodes:Union[dict,None]=None)->None:
        """Constructor."""
        self.nodes={} if dict is None else nodes
        self.nodes_count={}
    def reset(self)->None:
        """Reset the nodes data."""
        self.nodes={}
        self.nodes_count={}
    def add_neighbors(self,start:str,end:str,value:Union[int,list,dict]=1)->None:
        """Add neighbors node."""
        if start not in self.nodes:
            self.nodes[start]={}
        if start not in self.nodes_count:
            self.nodes_count[start]=0
        self.nodes[start][end]=value
        self.nodes_count[start]=len(self.nodes[start])
    def get_neighbors(self,v:str)->list:
        """Get neighbors by name."""
        return self.nodes.get(v,[])
    def iterate_neighbors(self,neighbors)->tuple:
        """Iterator of neighbors data."""
        return neighbors.items() if isinstance(neighbors,(dict,defaultdict)) else neighbors
    def get_weight(self,data_or_weight)->int:
        """Get weight if data is not scalar."""
        return data_or_weight[0] if isinstance(data_or_weight,
            (list,set,tuple,np.ndarray)) else data_or_weight
#    def h(self,n:str)->int:
#        """Heuristic distance function."""
##        return {k:1 for k in self.nodes.keys()}.get(n,1)
#        return 1
    def find(self,start_node:str,stop_node:str)->list:
        """Traverse the graph from start to end."""
        open_list=set([start_node])
        closed_list=set([])
        g={start_node:0}
        parents={start_node:start_node}
        while len(open_list)>0:
            n=None
            for v in open_list:
#                if n is None or g[v]+self.h(v)<g[n]+self.h(n):
                if n is None or g[v]<g[n]:
                    n=v
            if n is None:
                return []
            if n==stop_node:
                reconst_path=[]
                while parents[n]!=n:
                    reconst_path.append(n)
                    n=parents[n]
                reconst_path.append(start_node)
                return reconst_path[::-1]
            neighbors=self.get_neighbors(n)
            for (m,data_or_weight) in self.iterate_neighbors(neighbors):
                weight=self.get_weight(data_or_weight)
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m]=n
                    g[m]=g[n]+weight
                else:
                    if g[m]>g[n]+weight:
                        g[m]=g[n]+weight
                        parents[m]=n
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
        return []
