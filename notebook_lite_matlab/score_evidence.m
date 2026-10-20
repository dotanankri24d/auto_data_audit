function result = score_evidence(ids,y,classes,q,support,kind)
% Ranking definitions shared with notebook_lite; scores are not probabilities.
ids=string(ids(:)); y=string(y(:)); classes=string(classes(:));
n=numel(ids); c=numel(classes);
assert(isequal(size(q),[n c]),'audit:Evidence','Wrong evidence dimensions.');
issue=nan(n,1); overlap=issue; alt=strings(n,1); a=alt; b=alt;
alt(:)=missing; a(:)=missing; b(:)=missing;
opID=strings(0,1); opA=opID; opB=opID; opScore=zeros(0,1);
for i=1:n
    if c<2 || any(~isfinite(q(i,:))), continue; end
    [~,order]=sortrows([-q(i,:)' (1:c)'],[1 2]);
    original=find(classes==y(i),1); assert(~isempty(original),'audit:Label','Unknown original label.');
    other=order(find(order~=original,1));
    issue(i)=max(0,q(i,other)-q(i,original)); alt(i)=classes(other);
    a(i)=classes(order(1)); b(i)=classes(order(2));
    overlap(i)=2*min(q(i,order(1:2)));
    pairs=sort([repmat(original,c-1,1) setdiff((1:c)',original)],2);
    pairs=unique([pairs; sort(order(1:2))'],'rows');
    for k=1:size(pairs,1)
        opID(end+1,1)=ids(i); opA(end+1,1)=classes(pairs(k,1)); opB(end+1,1)=classes(pairs(k,2));
        opScore(end+1,1)=2*min(q(i,pairs(k,:)));
    end
end
result.samples=table(ids,y,logical(support(:)),repmat(string(kind),n,1),issue,alt,overlap,a,b, ...
    'VariableNames',{'sample_id','original_label','sufficient_support','support_kind','label_issue_score','alternative_label','max_overlap_score','overlap_class_a','overlap_class_b'});
result.overlaps=table(opID,opA,opB,opScore,'VariableNames',{'sample_id','class_a','class_b','pair_overlap'});
result.class_evidence=table(repelem(ids,c),repmat(classes,n,1),reshape(q',[],1),repmat(string(kind),n*c,1), ...
    'VariableNames',{'sample_id','class_label','class_score','source'});
end
