function detail=inspect_sample(result,id)
id=string(id); detail.sample=result.samples(result.samples.sample_id==id,:);
assert(height(detail.sample)==1,'audit:IDs','Sample ID not found.');
for name=["class_evidence","neighbors","oof","overlaps"]
    if isfield(result,name)
        t=result.(name); key="sample_id"; if name=="neighbors", key="query_sample_id"; end
        detail.(name)=t(t.(key)==id,:);
    end
end
end
