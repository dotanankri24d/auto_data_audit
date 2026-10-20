function out=compare_methods(results)
% Align by explicit sample ID; average-tie percentile ranks, no score averaging.
names=string(fieldnames(results)); ids=results.(names(1)).samples.sample_id;
out=table(ids,'VariableNames',{'sample_id'});
for name=names'
    t=results.(name).samples; [found,ix]=ismember(ids,t.sample_id);
    assert(all(found)&&height(t)==numel(ids),'audit:Alignment','Methods must contain the same IDs.');
    for score=["label_issue_score","max_overlap_score"]
        v=t.(score)(ix); rank=nan(size(v)); valid=isfinite(v); rank(valid)=tiedrank(v(valid))/sum(valid);
        out.(name+"_"+score+"_rank")=rank;
    end
end
end
