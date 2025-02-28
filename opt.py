import torch 
import math 

class OPT(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 loss_fn,
                 lr=1e-4, 
                 param_lr=None,
                 mode='optv1',
                 clip_value=1.0, 
                 device=None,
                 **kwargs):

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:  
            self.device = device  
        
        self.params = list(params)
        self.lr = lr
        self.param_lr = lr if param_lr is None else param_lr
        self.mode = mode.lower()
        self.clip_value = clip_value
        self.loss_fn = loss_fn
        self.a = None
        self.b = None
        self.alpha = None
        self.margin = None

        try:
            self.a = loss_fn.a 
            self.b = loss_fn.b 
            self.alpha = loss_fn.alpha 
            self.margin = loss_fn.margin
        except:
            print('AUCMLoss is not found!')

        # init 
        self.steps = 0            # total optimization steps
    
        # assert self.mode in ['optv1', 'optv2'], "Keyword is not found in [`optv1`, `optv2`]!"
       
        if self.a is not None and self.b is not None:
           self.params = self.params + [self.a, self.b]

        self.defaults = dict(lr=self.lr, 
                             param_lr=self.param_lr, 
                             margin=self.margin, 
                             a=self.a, 
                             b=self.b,
                             alpha=self.alpha)
        
        super(OPT, self).__init__(self.params, self.defaults)
         
    def __setstate__(self, state):
        super(OPT, self).__setstate__(state)
        for group in self.param_groups:
          if self.mode == 'optv1':
             group.setdefault('optv1', False)
          elif self.mode == 'optv2':
             group.setdefault('optv2', False)
          else:
             NotImplementedError    
    
    @property    
    def optim_step(self):
        r"""Return the number of optimization steps."""
        return self.steps
    
    @torch.no_grad()
    def step(self, lr=None, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            self.lr =  group['lr'] if lr is None else lr
            print(f"lr={self.lr}")
            self.param_lr = group['param_lr']
            
            m = group['margin'] 
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                # d_p = torch.clamp(p.grad.data , -self.clip_value, self.clip_value)
                d_p = p.grad.data
                if len(d_p.shape) > 1: 
                    p.data = p.data - group['lr']*d_p
                else:   
                    p.data = p.data - group['param_lr']*d_p
            
            if alpha is not None: 
               if alpha.grad is not None: 
                #   alpha.data = alpha.data + group['lr']*(2*(m + b.data - a.data)-2*alpha.data)
                  alpha.data = alpha.data + group['param_lr']*alpha.grad 
                  alpha.data  = torch.clamp(alpha.data,  0, 999)

        self.steps += 1
        return loss
