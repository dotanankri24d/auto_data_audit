function fig=audit_scatter(coordinates,samples,color,selected)
% Native figures support data tips, zoom, pan, and brushing.
if nargin<3, color="original_label"; end
if nargin<4, selected=strings(0,1); end
[found,ix]=ismember(samples.sample_id,coordinates.sample_id);
assert(all(found),'audit:Alignment','Coordinates missing sample IDs.');
z=coordinates{ix,2:end}; x=z(:,1); y=z(:,min(2,size(z,2)));
fig=figure; ax=axes(fig); hold(ax,'on'); v=samples.(color);
if isnumeric(v)
    h=scatter(ax,x,y,28,v,'filled'); colorbar(ax);
    h.DataTipTemplate.DataTipRows(end+1)=dataTipTextRow('sample_id',samples.sample_id);
else
    classes=unique(string(v),'stable'); palette=lines(numel(classes));
    for j=1:numel(classes)
        keep=string(v)==classes(j); h=scatter(ax,x(keep),y(keep),28,palette(j,:),'filled','DisplayName',classes(j));
        h.DataTipTemplate.DataTipRows(end+1)=dataTipTextRow('sample_id',samples.sample_id(keep));
    end
    legend(ax,'Location','best');
end
keep=ismember(samples.sample_id,string(selected)); scatter(ax,x(keep),y(keep),100,'k','LineWidth',1.5,'HandleVisibility','off');
xlabel(ax,coordinates.Properties.VariableNames{2},'Interpreter','none');
ylabel(ax,coordinates.Properties.VariableNames{1+min(2,size(z,2))},'Interpreter','none');
title(ax,color,'Interpreter','none'); grid(ax,'on');
end
