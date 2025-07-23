# Finding minimum t
## Outline
This file finds the minimum time elapsed such that the difference between the startup flow and its steady state for 
each core model are below a certain tolerance.

## Upgrades
It became apparent that the minimum t to reach steady state are different for different length scales. Intuitively, this
makes sense and therefore, for each L in global variables ('L-list'), the minimum t to reach the steady state will be 
computed. w
