function result = neighbors(X,y,ids,o)
% Exact batched neighbors; self excluded by row, not feature equality.
arguments
    X table
    y
    ids
    o.Groups = []
    o.Embedding = struct('kind','identity')
    o.Metric = "euclidean"
    o.NumNeighbors (1,1) double {mustBeInteger,mustBePositive} = 30
    o.Weighting (1,1) string {mustBeMember(o.Weighting,["uniform","inverse_distance"])} = "uniform"
    o.RadiusMode (1,1) string {mustBeMember(o.RadiusMode,["auto","fixed","none"])} = "auto"
    o.RadiusQuantile (1,1) double {mustBeGreaterThanOrEqual(o.RadiusQuantile,0),mustBeLessThanOrEqual(o.RadiusQuantile,1)} = .95
    o.MaxRadius (1,1) double = NaN
    o.MinNeighbors (1,1) double {mustBeInteger,mustBePositive} = 10
    o.MinEffectiveNeighbors (1,1) double {mustBePositive,mustBeFinite} = 5
    o.ExcludeSameGroup (1,1) logical = false
    o.DistanceEpsilon (1,1) double {mustBePositive,mustBeFinite} = 1e-8
    o.BatchSize (1,1) double {mustBeInteger,mustBePositive} = 128
    o.SaveEvidence (1,1) logical = true
end
[x,y,ids,classes,g]=audit_inputs(X,y,ids,o.Groups); n=size(x,1);
assert(~o.ExcludeSameGroup||~isempty(g),'audit:Groups','Group exclusion requires groups.');
if o.RadiusMode=="fixed", assert(isfinite(o.MaxRadius)&&o.MaxRadius>=0,'audit:Radius','Fixed radius must be finite and nonnegative.'); end
transform=fit_representation(x,o.Embedding); z=transform(x);
nearest=cell(n,1); distances=nearest; kth=nan(n,1);
for start=1:o.BatchSize:n
    stop=min(n,start+o.BatchSize-1);
    if isa(o.Metric,'function_handle')
        d=zeros(stop-start+1,n);
        for i=start:stop
            for j=1:n
                v=o.Metric(z(i,:),z(j,:));
                assert(isnumeric(v)&&isreal(v)&&isscalar(v)&&isfinite(v)&&v>=0,'audit:Distance','Distance must be finite and nonnegative.');
                d(i-start+1,j)=v;
            end
        end
    else
        d=pdist2(z(start:stop,:),z,char(o.Metric));
        assert(all(isfinite(d)&d>=0,'all'),'audit:Distance','Metric returned invalid distances.');
    end
    for i=start:stop
        row=d(i-start+1,:); row(i)=inf;
        if o.ExcludeSameGroup, row(g==g(i))=inf; end
        eligible=find(isfinite(row));
        sorted=sortrows([row(eligible)' eligible'],[1 2]);
        sorted=sorted(1:min(o.NumNeighbors,size(sorted,1)),:);
        nearest{i}=sorted(:,2); distances{i}=sorted(:,1);
        if size(sorted,1)==o.NumNeighbors, kth(i)=sorted(end,1); end
    end
end
radius=NaN;
if o.RadiusMode=="auto"
    assert(all(isfinite(kth)),'audit:Radius','Too few eligible neighbors for auto radius; choose none or reduce k.');
    % NumPy's linear quantile definition (MATLAB quantile uses another convention).
    sorted=sort(kth); pos=1+(n-1)*o.RadiusQuantile;
    radius=sorted(floor(pos))+(pos-floor(pos))*(sorted(ceil(pos))-sorted(floor(pos)));
elseif o.RadiusMode=="fixed", radius=o.MaxRadius;
end
q=nan(n,numel(classes)); support=false(n,1); diag=nan(n,7);
query=strings(0,1); ref=query; rank=zeros(0,1); dist=rank; weight=rank; label=query;
for i=1:n
    ix=nearest{i}; ds=distances{i};
    if ~isnan(radius), keep=ds<=radius; ix=ix(keep); ds=ds(keep); end
    w=ones(size(ds)); if o.Weighting=="inverse_distance", w=1./max(ds,o.DistanceEpsilon); end
    effective=NaN; near=NaN; far=NaN; med=NaN; ng=NaN;
    if ~isempty(ix)
        effective=sum(w)^2/sum(w.^2); near=ds(1); far=ds(end); med=median(ds);
        for c=1:numel(classes), q(i,c)=sum(w(y(ix)==classes(c)))/sum(w); end
    end
    if ~isempty(g), ng=numel(unique(g(ix))); end
    support(i)=numel(ix)>=o.MinNeighbors && effective>=o.MinEffectiveNeighbors;
    diag(i,:)=[numel(ix) effective near far med radius ng];
    if o.SaveEvidence
        query=[query; repmat(ids(i),numel(ix),1)]; ref=[ref;ids(ix)]; rank=[rank;(1:numel(ix))'];
        dist=[dist;ds]; weight=[weight;w]; label=[label;y(ix)]; %#ok<AGROW>
    end
end
kind="local_neighbors"; if isnan(radius), kind="neighbor_count_and_weight_only"; end
result=score_evidence(ids,y,classes,q,support,kind);
result.samples=[result.samples array2table(diag,'VariableNames',{'neighbor_count','effective_neighbors','nearest_distance','max_distance','median_distance','radius_used','reference_group_count'})];
result.neighbors=table(query,ref,rank,dist,weight,label,'VariableNames',{'query_sample_id','neighbor_sample_id','rank','distance','weight','neighbor_label'});
result.coordinates=[table(ids,'VariableNames',{'sample_id'}) array2table(z,'VariableNames',cellstr("z"+(1:size(z,2))))];
result.run_info=struct('calibrated_radius',radius,'evidence_saved',o.SaveEvidence,'feature_names',{X.Properties.VariableNames});
end
