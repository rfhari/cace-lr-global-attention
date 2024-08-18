# Setup:
* Install cace as described here: https://github.com/BingqingCheng/cace/tree/main 
* Install Fast Attention Via positive Orthogonal Random features approach (FAVOR+) as described here: https://github.com/lucidrains/performer-pytorch 

# Major Changes:
* Nonlocal corrections to "q" vectors using global attention: https://github.com/rfhari/cace-lr-global-attention/blob/92de51efe1196b47423651d5db143d7914526370/cace/cace/modules/atomwise.py#L164
* Implementation of favor+ on each graph (of each batch): [cace/cace/modules/global_attention.py](url)
