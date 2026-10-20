function predictor=train_softmax(x,y,c,seed) %#ok<INUSD>
% L2 multinomial logistic regression, unpenalized intercept, C=1.
% Deterministic BFGS avoids an Optimization Toolbox dependency.
x=[ones(size(x,1),1) x]; p=size(x,2); w=zeros(p*c,1); H=eye(p*c);
[f,g]=objective(w,x,y,c); converged=false;
for iteration=1:1000
    if norm(g,inf)<1e-7, converged=true; break; end
    direction=-H*g;
    if g'*direction>=0, H=eye(p*c); direction=-g; end
    step=1;
    for search=1:60
        candidate=w+step*direction; [next,grad]=objective(candidate,x,y,c);
        if next<=f+1e-4*step*(g'*direction), break; end
        step=step/2;
    end
    s=candidate-w; delta=grad-g; curvature=delta'*s;
    if curvature>1e-12*norm(s)*norm(delta)
        rho=1/curvature; V=eye(p*c)-rho*s*delta'; H=V*H*V'+rho*(s*s');
    end
    w=candidate; f=next; g=grad;
end
assert(converged||norm(g,inf)<1e-6,'audit:Convergence','Logistic fit did not converge; rescale features or supply an estimator.');
weights=reshape(w,p,c); predictor=@(new)probabilities([ones(size(new,1),1) new]*weights);
end
function [loss,g]=objective(w,x,y,c)
W=reshape(w,size(x,2),c); scores=x*W; m=max(scores,[],2); shifted=scores-m;
p=exp(shifted); p=p./sum(p,2); n=size(x,1); ix=sub2ind([n c],(1:n)',y);
penalty=W; penalty(1,:)=0;
loss=mean(log(sum(exp(shifted),2))-shifted(ix))+.5*sum(penalty.^2,'all')/n;
p(ix)=p(ix)-1; gradient=(x'*p+penalty)/n; g=gradient(:);
end
function p=probabilities(scores)
p=exp(scores-max(scores,[],2)); p=p./sum(p,2);
end
