%% Data audit in MATLAB
% A standalone MATLAB port of notebook_lite. Open analysis.mlx in Live Editor
% and Run All. Scores measure evidence, not probabilities of incorrect labels.
% Demo data are synthetic. Thresholds require calibration; NaN flags abstain.
root=fileparts(mfilename('fullpath'));
if isempty(root), root=pwd; end
addpath(root);

%% 1. Load CSV and choose explicit features
% IDs are imported as strings, preserving leading zeroes. Clean missing values
% and encode categorical features explicitly. Never include IDs or metadata in X.
CSV_PATH=fullfile(root,'demo.csv');
ID="sample_id"; LABEL="label"; GROUP="";
FEATURES=["feature_0","feature_1","feature_2","feature_3"];
frame=load_csv(CSV_PATH,ID);
ids=string(frame.(ID)); y=string(frame.(LABEL)); X=frame(:,FEATURES);
groups=[]; if strlength(GROUP)>0, groups=frame.(GROUP); end
metadata=frame(:,setdiff(string(frame.Properties.VariableNames),[ID LABEL FEATURES],'stable'));
disp(head(frame)); disp(groupsummary(frame,LABEL)); summary(X);
% Excel alternative: detectImportOptions + setvartype(ID,'string') + readtable.

%% 2. Choose representation
% Neighbors fit all reference rows; OOF refits inside every training fold.
% Custom fitter: @(train) a function @(new) returning a finite numeric matrix.
embedding=struct('kind','pca','components',2,'standardize',true,'whiten',false);
% embedding=struct('kind','identity');
% embedding=struct('kind','standardize');
% embedding=struct('kind','pca','components',.95,'whiten',true);
% embedding=struct('kind','pipeline','steps',{{struct('kind','standardize'), ...
%     struct('kind','pca','components',2,'standardize',false)}});
% embedding=struct('kind','autoencoder','components',2,'epochs',50,'seed',42);
% Autoencoder requires Deep Learning Toolbox and uses MATLAB's native backend.

%% 3. Neighbor evidence
% RadiusMode: auto, fixed (MaxRadius), or none. Self is always excluded.
% ExcludeSameGroup controls neighbor exclusion separately from OOF isolation.
neighbor_result=neighbors(X,y,ids,Groups=groups,Embedding=embedding, ...
    NumNeighbors=20,Weighting="inverse_distance",RadiusMode="auto", ...
    RadiusQuantile=.95,MinNeighbors=8,MinEffectiveNeighbors=4,BatchSize=128);
ranked=sortrows(neighbor_result.samples,'label_issue_score','descend');
disp(head(ranked,10));
% Custom metric: Metric=@(a,b)sum(abs(a-b))

%% 4. Repeated out-of-fold predictions
% Each test sample and its group are absent from that fold's training data.
% The default is L2 multinomial logistic regression (C=1). Custom Estimator:
% @(trainX,trainCodes,numClasses,seed) returning @(testX) probabilities.
% Columns must follow training codes 1:numClasses, with rows summing to one.
oof_result=oof(X,y,ids,Groups=groups,Embedding=embedding, ...
    NumSplits=5,NumRepeats=3,RandomSeed=42,MinTrainPerClass=2);
results=struct('neighbors',neighbor_result,'oof',oof_result);
disp(head(sortrows(oof_result.samples,'label_issue_score','descend'),10));
% Exact split IDs are saved in run_info. Repeated dispersion is descriptive,
% not a confidence interval; one repeat/identical partitions gives NaN.

%% 5. Thresholds, filtering, metadata and review
% issue=max(0,q(alternative)-q(original)); overlap(a,b)=2*min(q(a),q(b)).
% Flags are doubles: 1 review, 0 below threshold, NaN abstain.
method="neighbors"; result=results.(method); samples=result.samples;
flags=review_flags(samples,IssueThreshold=.4,OverlapThreshold=.6);
% OOF may also use MinAlternativeAgreement=.8.
findings=[samples flags];
[aligned,order]=ismember(findings.sample_id,ids); assert(all(aligned));
for column=string(metadata.Properties.VariableNames)
    findings.("metadata_"+column)=metadata.(column)(order,:);
end
filtered=sortrows(findings(findings.label_issue==1,:),'label_issue_score','descend');
selected_ids=filtered.sample_id(1:min(5,height(filtered)));
disp(head(filtered,20));

%% 6. Interactive views and distributions
% Native data tips contain IDs. Use zoom, pan and brushing in figure windows.
% A display-only PCA can be fitted separately and never changes OOF evidence.
coordinates=neighbor_result.coordinates;
class_figure=audit_scatter(coordinates,findings,"original_label",selected_ids);
score_figure=audit_scatter(coordinates,findings,"label_issue_score",selected_ids);
figure; histogram(findings.label_issue_score,30); xlabel('Label issue score'); ylabel('Count');

%% 7. Inspect samples and preserve review notes separately
sample_id=ids(1); if ~isempty(selected_ids), sample_id=selected_ids(1); end
disp(frame(ids==sample_id,:)); detail=inspect_sample(result,sample_id); disp(detail);
local_neighbors=neighbor_result.neighbors(neighbor_result.neighbors.query_sample_id==sample_id,:);
[~,rows]=ismember(local_neighbors.neighbor_sample_id,ids); disp(frame(rows,:));
class_scores=result.class_evidence(result.class_evidence.sample_id==sample_id,:);
figure; bar(categorical(class_scores.class_label),class_scores.class_score); ylabel('Class evidence');
oof_detail=inspect_sample(oof_result,sample_id); disp(oof_detail.oof);
reviews=table('Size',[0 5],'VariableTypes',repmat("string",1,5), ...
    'VariableNames',{'sample_id','method','review_status','proposed_label','note'});
% reviews(end+1,:)={sample_id,method,"confirmed","class_1","Reviewed source"};

%% 8. Class/metadata summaries, pair overlap and method comparison
% Pair averages concern recorded original-label pairs plus the leading pair.
disp(audit_summary(findings));
if width(metadata)>0
    disp(audit_summary(findings,"metadata_"+string(metadata.Properties.VariableNames{1})));
end
pairs=groupsummary(result.overlaps,{'class_a','class_b'},'mean','pair_overlap'); disp(pairs);
if ~isempty(pairs), figure; heatmap(pairs,'class_a','class_b','ColorVariable','mean_pair_overlap'); end
comparison=compare_methods(results); disp(head(comparison));
neighbor_flags=review_flags(neighbor_result.samples); oof_flags=review_flags(oof_result.samples);
disp(table(ids,neighbor_flags.label_issue,oof_flags.label_issue, ...
    'VariableNames',{'sample_id','neighbor_issue','oof_issue'}));

%% 9. Export full, filtered and selected evidence
% CSV includes explicit IDs. JSON stores fold provenance. FIG preserves native
% interactive plots; PNG is portable. Save the executed Live Script in MATLAB.
EXPORT=fullfile(root,'exports');
export_tables(neighbor_result,fullfile(EXPORT,'neighbors'));
export_tables(oof_result,fullfile(EXPORT,'oof'));
writetable(findings,fullfile(EXPORT,'findings.csv'));
writetable(filtered,fullfile(EXPORT,'filtered.csv'));
writetable(findings(ismember(findings.sample_id,selected_ids),:),fullfile(EXPORT,'selected.csv'));
writetable(reviews,fullfile(EXPORT,'reviews.csv'));
writetable(frame,fullfile(EXPORT,'source_snapshot.csv'));
savefig(class_figure,fullfile(EXPORT,'scatter.fig'));
exportgraphics(class_figure,fullfile(EXPORT,'scatter.png'));

%% 10. Extend with ordinary MATLAB functions
% Copy this entire folder anywhere. No parent package or Python is needed.
% Reanalyzing a subset creates a new reference population; filtering does not.
% A training-scoped centering fitter is supplied below as an example.
disp(findings(findings.label_issue_score>=.7,:));
% custom=neighbors(X,y,ids,Embedding=@center_only,NumNeighbors=10, ...
%     RadiusMode="none",MinNeighbors=5);
