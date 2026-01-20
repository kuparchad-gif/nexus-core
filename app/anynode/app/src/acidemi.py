from typing import Dict, Any
from .metatron import qvec

def choose(query: str, lattice, facet_weights: Dict[str,float], k:int=5):
  seeds=[sid for sid,_ in lattice.vec_search(qvec(query),k)]
  ranked=lattice.geo(seeds,facet_weights,k=k)
  rat=[{'id':nid,'score':sc,'facets':fac} for (nid,sc,fac) in ranked]
  return {'choice': rat[0] if rat else None, 'rationale': rat, 'seeds':seeds}
