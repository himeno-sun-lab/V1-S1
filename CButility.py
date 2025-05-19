import nest
from nest import topology as tp
#import nest.topology as tp

def create_layer(rows=32, columns=32, center=[0., 0.], extent=[1024., 1024.], elements='iaf_cond_exp', edge_wrap=False):
    
    configuration = {'rows': rows,
                     'columns': columns,
                     'extent': extent,
                     'elements': elements,
                     'edge_wrap': edge_wrap}
    layer = tp.CreateLayer(configuration)
    return layer
