function out=audit_summary(samples,by)
if nargin<2, by="original_label"; end
values=string(samples.(by)); keys=unique(values,'stable'); rows=cell(numel(keys),1);
for i=1:numel(keys)
    if ismissing(keys(i)), keep=ismissing(values); else, keep=values==keys(i); end
    t=samples(keep,:);
    row=table(keys(i),height(t),sum(t.sufficient_support),mean(t.label_issue_score,'omitnan'),mean(t.max_overlap_score,'omitnan'), ...
        'VariableNames',{'category','count','supported','mean_issue','mean_overlap'});
    for flag=["label_issue","overlap"]
        if ismember(flag,string(t.Properties.VariableNames))
            v=t.(flag); total=sum(v,'omitnan'); if all(isnan(v)), total=NaN; end
            row.(flag+"_count")=total; row.(flag+"_abstentions")=sum(isnan(v));
        end
    end
    rows{i}=row;
end
out=vertcat(rows{:});
end
