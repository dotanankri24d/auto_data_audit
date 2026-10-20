function frame=load_csv(path,idColumn)
% Force textual IDs before import so leading zeroes survive.
opts=detectImportOptions(path,'VariableNamingRule','preserve');
assert(ismember(idColumn,string(opts.VariableNames)),'audit:IDs','ID column not found.');
opts=setvartype(opts,char(idColumn),'string'); frame=readtable(path,opts);
ids=string(frame.(idColumn));
assert(~any(ismissing(ids)|strlength(ids)==0)&&numel(unique(ids))==height(frame),'audit:IDs','IDs must be unique and present.');
end
