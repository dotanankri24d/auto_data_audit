function [x,y,ids,classes,g] = audit_inputs(X,y,ids,g)
% Explicit IDs accompany every row; tables retain feature names and order.
assert(istable(X) && width(X)>0 && height(X)>0,'audit:Input','X must be a nonempty numeric table.');
x = X{:,:};
assert(isnumeric(x) && isreal(x) && all(isfinite(x),'all'),'audit:Input','Features must be finite real numbers.');
x = double(x); y = string(y(:)); ids = string(ids(:)); g = string(g(:));
n = height(X);
assert(numel(y)==n && numel(ids)==n && (isempty(g)||numel(g)==n),'audit:Alignment','Labels, IDs and groups must match X row order.');
assert(~any(ismissing(ids)|strlength(ids)==0) && numel(unique(ids))==n,'audit:IDs','IDs must be present and unique.');
assert(~any(ismissing(y)|strlength(y)==0) && ~any(ismissing(g)|strlength(g)==0),'audit:Labels','Labels and groups must be present.');
classes = unique(y,'stable');
end
