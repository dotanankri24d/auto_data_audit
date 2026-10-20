function checks
% Run: cd notebook_lite_matlab; checks
root=fileparts(mfilename('fullpath')); addpath(root);
t=load_csv(fullfile(root,'demo.csv'),'sample_id');
assert(t.sample_id(1)=="00000",'Leading zeroes were lost.');
X=t(:,"feature_"+(0:3)); y=string(t.label); ids=t.sample_id;
X{2,:}=X{1,:};
r=neighbors(X,y,ids,NumNeighbors=5,RadiusMode="none",MinNeighbors=3);
assert(~any(r.neighbors.query_sample_id==r.neighbors.neighbor_sample_id));
assert(any(r.neighbors.query_sample_id==ids(1)&r.neighbors.neighbor_sample_id==ids(2)&r.neighbors.distance==0));
r=neighbors(X,y,ids,NumNeighbors=5,RadiusMode="fixed",MaxRadius=0);
f=review_flags(r.samples); assert(all(isnan(f{:,:}),'all'));
expect_error(@()neighbors(X(1:3,:),y(1:3),ids(1:3),NumNeighbors=5),'audit:Radius');
bad=X; bad{1,1}=NaN; expect_error(@()neighbors(bad,y,ids),'audit:Input');
badids=ids; badids(2)=ids(1); expect_error(@()neighbors(X,y,badids),'audit:IDs');
% Fold-local fit count and group isolation; deliberately track fitter inputs.
fitted=[]; groups="group_"+ceil((1:height(X))'/2); state=rng;
r=oof(X,y,ids,Groups=groups,Embedding=@record_fit,NumSplits=3,NumRepeats=2);
assert(isequal(rng,state),'OOF changed caller random state.');
assert(numel(fitted)==6&&all(fitted>0&fitted<height(X)));
for s=r.run_info.splits
    [~,tr]=ismember(s.train_ids,ids); [~,te]=ismember(s.test_ids,ids);
    assert(isempty(intersect(groups(tr),groups(te))));
    assert(isempty(intersect(s.train_ids,s.test_ids)));
end
sums=groupsummary(r.class_evidence,'sample_id','sum','class_score');
assert(all(abs(sums.sum_class_score-1)<1e-12));
r=oof(X,y,ids,NumSplits=3,NumRepeats=1);
assert(all(isnan(r.samples.label_issue_std)));
% Direct score formula and stable class tie checks.
s=score_evidence(["a";"b"],["A";"B"],["A";"B";"C"],[.1 .6 .3;.5 .25 .25],[true;true],'test');
assert(max(abs(s.samples.label_issue_score-[.5;.25]))<1e-12);
assert(max(abs(s.samples.max_overlap_score-[.6;.5]))<1e-12);
assert(isequal(s.samples.alternative_label,["B";"A"]));
% Golden fixtures produced by the original Python implementation, identity
% embedding. Covers all radii, weights, pair evidence, diagnostics, and ties.
original=load_csv(fullfile(root,'demo.csv'),'sample_id'); X=original(:,"feature_"+(0:3));
for mode=["none","fixed","auto"]
    for weighting=["uniform","inverse_distance"]
        r=neighbors(X,string(original.label),original.sample_id,NumNeighbors=20, ...
            RadiusMode=mode,MaxRadius=1.5,Weighting=weighting,MinNeighbors=8,MinEffectiveNeighbors=4);
        fixture=readtable(fullfile(root,'reference',mode+"_"+weighting+".csv"),'TextType','string');
        for column=["label_issue_score","max_overlap_score","neighbor_count","effective_neighbors","nearest_distance","max_distance","median_distance","radius_used"]
            a=r.samples.(column); b=fixture.(column);
            assert(isequal(isnan(a),isnan(b))); valid=isfinite(a);
            assert(all(abs(a(valid)-b(valid))<1e-10),'Python neighbor reference mismatch.');
        end
        assert(isequal(r.samples.sufficient_support,logical(fixture.sufficient_support)));
    end
end
fprintf('MATLAB data-audit checks passed.\n');
    function transform=record_fit(train)
        fitted(end+1)=size(train,1); transform=fit_representation(train,struct('kind','standardize'));
    end
end
function expect_error(action,identifier)
try
    action();
catch exception
    assert(strcmp(exception.identifier,identifier),'Unexpected error: %s',exception.message); return;
end
error('audit:Check','Expected error %s was not raised.',identifier);
end
