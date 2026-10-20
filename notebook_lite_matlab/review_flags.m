function flags=review_flags(samples,o)
% Numeric flags: 1=review, 0=below threshold, NaN=abstain.
arguments
    samples table
    o.IssueThreshold (1,1) double {mustBeGreaterThanOrEqual(o.IssueThreshold,0),mustBeLessThanOrEqual(o.IssueThreshold,1)} = .5
    o.OverlapThreshold (1,1) double {mustBeGreaterThanOrEqual(o.OverlapThreshold,0),mustBeLessThanOrEqual(o.OverlapThreshold,1)} = .6
    o.RequireSupport (1,1) logical = true
    o.MinAlternativeAgreement = []
end
valid=true(height(samples),1);
if o.RequireSupport, valid=valid & samples.sufficient_support; end
if ~isempty(o.MinAlternativeAgreement)
    t=o.MinAlternativeAgreement; assert(isscalar(t)&&isfinite(t)&&t>=0&&t<=1,'audit:Threshold','Invalid agreement threshold.');
    if ismember('alternative_agreement',samples.Properties.VariableNames), valid=valid & samples.alternative_agreement>=t; else, valid(:)=false; end
end
a=double(samples.label_issue_score>=o.IssueThreshold); b=double(samples.max_overlap_score>=o.OverlapThreshold);
a(~valid|~isfinite(samples.label_issue_score))=NaN; b(~valid|~isfinite(samples.max_overlap_score))=NaN;
flags=table(a,b,'VariableNames',{'label_issue','overlap'});
end
