function result=oof(X,y,ids,o)
% Repeated stratified folds. Each representation/model fits training rows only.
% Estimator: @(trainX,trainCodes,numClasses,seed) predictor, where
% predictor(newX) returns probabilities in first-seen class order.
arguments
    X table
    y
    ids
    o.Groups = []
    o.Embedding = struct('kind','identity')
    o.Estimator = @train_softmax
    o.NumSplits (1,1) double {mustBeInteger,mustBeGreaterThanOrEqual(o.NumSplits,2)} = 5
    o.NumRepeats (1,1) double {mustBeInteger,mustBePositive} = 3
    o.RandomSeed (1,1) double {mustBeInteger,mustBeNonnegative} = 42
    o.MinTrainPerClass (1,1) double {mustBeInteger,mustBePositive} = 2
    o.MinValidRepeats = []
    o.SaveEvidence (1,1) logical = true
end
[x,y,ids,classes,g]=audit_inputs(X,y,ids,o.Groups); n=size(x,1); c=numel(classes);
required=o.MinValidRepeats; if isempty(required), required=o.NumRepeats; end
assert(isscalar(required)&&isfinite(required)&&required==fix(required)&&required>=1&&required<=o.NumRepeats,'audit:Repeats','Invalid repeat requirement.');
if c<2
    result=score_evidence(ids,y,classes,nan(n,c),false(n,1),'prediction_coverage'); return;
end
[~,codes]=ismember(y,classes); counts=accumarray(codes,1,[c 1]);
assert(all(counts>=o.NumSplits),'audit:Folds','Every class needs at least NumSplits rows.');
state=rng; cleanup=onCleanup(@()rng(state));
assignments=zeros(n,o.NumRepeats); splits=struct('repeat_id',{},'fold_id',{},'seed',{},'train_ids',{},'test_ids',{},'train_class_counts',{});
% Validate all splits before fitting any model.
for r=1:o.NumRepeats
    seed=o.RandomSeed+r-1; rng(seed,'twister');
    fold=stratify(codes,g,c,o.NumSplits); assignments(:,r)=fold;
    for f=1:o.NumSplits
        train=find(fold~=f); test=find(fold==f); counts=accumarray(codes(train),1,[c 1]);
        assert(~isempty(test)&&all(counts>=o.MinTrainPerClass),'audit:Folds','Empty test fold or insufficient training class support; reduce folds/change groups.');
        if ~isempty(g), assert(isempty(intersect(g(train),g(test))),'audit:Leakage','Group leakage.'); end
        splits(end+1)=struct('repeat_id',r,'fold_id',f,'seed',seed,'train_ids',ids(train),'test_ids',ids(test),'train_class_counts',counts); %#ok<AGROW>
    end
end
probs=nan(n,c,o.NumRepeats); evID=strings(0,1); evRepeat=zeros(0,1); evFold=evRepeat; evClass=evID; evScore=evRepeat;
for k=1:numel(splits)
    s=splits(k); train=find(assignments(:,s.repeat_id)~=s.fold_id); test=find(assignments(:,s.repeat_id)==s.fold_id);
    transform=fit_representation(x(train,:),o.Embedding);
    predictor=o.Estimator(transform(x(train,:)),codes(train),c,s.seed);
    assert(isa(predictor,'function_handle'),'audit:Estimator','Estimator must return a prediction function.');
    q=predictor(transform(x(test,:)));
    assert(isnumeric(q)&&isreal(q)&&isequal(size(q),[numel(test) c])&&all(isfinite(q)&q>=0&q<=1,'all')&&all(abs(sum(q,2)-1)<=1e-6),'audit:Prediction','Predictor must return normalized probabilities in class order.');
    probs(test,:,s.repeat_id)=q;
    if o.SaveEvidence
        evID=[evID;repelem(ids(test),c)]; evRepeat=[evRepeat;repmat(s.repeat_id,numel(test)*c,1)];
        evFold=[evFold;repmat(s.fold_id,numel(test)*c,1)]; evClass=[evClass;repmat(classes,numel(test),1)]; evScore=[evScore;reshape(q',[],1)]; %#ok<AGROW>
    end
end
coverage=sum(reshape(all(isfinite(probs),2),n,o.NumRepeats),2);
result=score_evidence(ids,y,classes,mean(probs,3),coverage>=required,'prediction_coverage');
canonical=zeros(size(assignments));
for r=1:o.NumRepeats
    map=zeros(o.NumSplits,1); next=0;
    for i=1:n
        f=assignments(i,r); if map(f)==0, next=next+1; map(f)=next; end
        canonical(i,r)=map(f);
    end
end
distinct=size(unique(canonical','rows'),1); issue=nan(n,o.NumRepeats); overlap=issue; alt=strings(n,o.NumRepeats);
for r=1:o.NumRepeats
    scored=score_evidence(ids,y,classes,probs(:,:,r),true(n,1),'prediction_coverage');
    issue(:,r)=scored.samples.label_issue_score; overlap(:,r)=scored.samples.max_overlap_score; alt(:,r)=scored.samples.alternative_label;
end
issueStd=nan(n,1); overlapStd=issueStd; agreement=issueStd;
if distinct>1
    issueStd=std(issue,0,2); overlapStd=std(overlap,0,2); agreement=mean(alt==result.samples.alternative_label,2);
end
result.samples=[result.samples table(coverage,issueStd,overlapStd,agreement,'VariableNames',{'valid_repeats','label_issue_std','overlap_std','alternative_agreement'})];
result.oof=table(evID,evRepeat,evFold,evClass,evScore,'VariableNames',{'sample_id','repeat_id','fold_id','class_label','class_score'});
result.run_info=struct('splits',splits,'distinct_split_repeats',distinct,'evidence_saved',o.SaveEvidence, ...
    'stability_note','Dispersion is not a confidence interval; unavailable for identical partitions.', ...
    'feature_names',{X.Properties.VariableNames});
end
function fold=stratify(codes,groups,c,k)
n=numel(codes); fold=zeros(n,1);
if isempty(groups)
    offset=0;
    for j=1:c
        ix=find(codes==j); ix=ix(randperm(numel(ix)));
        fold(ix)=mod((0:numel(ix)-1)+offset,k)+1; offset=mod(offset+numel(ix),k);
    end
else
    [ug,~,gi]=unique(groups); ng=numel(ug);
    assert(ng>=k,'audit:Groups','Fewer groups than folds.');
    gc=accumarray([gi codes],1,[ng c]); totals=sum(gc,1);
    % Shuffle ties, then allocate large/imbalanced groups first.
    shuffled=randperm(ng); [~,order]=sort(std(gc(shuffled,:),1,2),'descend'); order=shuffled(order);
    fc=zeros(k,c); sizes=zeros(k,1);
    for j=1:ng
        group=order(j); loss=inf(k,1);
        candidates=1:k; if j<=k, candidates=find(sizes==0)'; end
        for f=candidates
            candidate=fc; candidate(f,:)=candidate(f,:)+gc(group,:);
            loss(f)=mean(std(candidate./totals,1,1))+1e-9*sizes(f);
        end
        [~,f]=min(loss); fold(gi==group)=f; fc(f,:)=fc(f,:)+gc(group,:); sizes(f)=sizes(f)+sum(gc(group,:));
    end
end
end
