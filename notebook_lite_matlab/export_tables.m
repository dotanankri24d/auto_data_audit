function export_tables(result,directory)
% CSV for tables, JSON for split provenance; no custom result archive.
if ~isfolder(directory), mkdir(directory); end
for name=string(fieldnames(result))'
    value=result.(name);
    if istable(value), writetable(value,fullfile(directory,name+".csv"),'Encoding','UTF-8'); end
end
if isfield(result,'run_info')
    file=fopen(fullfile(directory,'run_info.json'),'w','n','UTF-8'); assert(file~=-1,'audit:Export','Cannot open JSON output.');
    cleanup=onCleanup(@()fclose(file)); fprintf(file,'%s',jsonencode(result.run_info,'PrettyPrint',true));
end
end
