#!/usr/bin/env python3

import numpy as np

from myalloy import misfit_volume as mf



# for each sample, compositions from 1 to N-1, and V

data1=np.array([
[   63/96, 11.64650686 ],
[   65/96, 11.59153841 ],

[   62/96, 11.67518402 ],
[   66/96, 11.56469966 ],

[  68/108, 11.72573906   ],
[  68/108, 11.72708519   ],
[  68/108, 11.72541852   ],

[  76/108,  11.51105252   ],
[  76/108,  11.51310370   ],
[  76/108,  11.50856667   ],
[  76/108,  11.51375185   ],
[  76/108,  11.50514074   ],

[  2/3, 11.61826447   ],
[  2/3, 11.61516906   ],
[  2/3, 11.61134444   ],

[  2/3, 11.61190000   ],
[  2/3, 11.61634444   ],
[  2/3, 11.61578889   ],

[  2/3, 11.60708519   ],
[  2/3, 11.61810370   ],
[  2/3, 11.61606667   ],

])



# uncertainty for data1

data2=np.zeros( data1.shape )



# central compostion, from 1 to N-1

cn=np.array([2/3])



mf.run_linear_reg_for_misfit(data1, data2, cn)
mf.check_linear_reg_results(data1, data2)
mf.check_misfit_precision_uncertainty(data1, data2, cn)


