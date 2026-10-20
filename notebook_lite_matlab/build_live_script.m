function build_live_script
% Regenerate the Live Script from its reviewable .m companion using MATLAB.
% This conversion utility is provided by MATLAB's Live Editor.
root=fileparts(mfilename('fullpath'));
matlab.internal.liveeditor.openAndConvert(fullfile(root,'analysis.m'),fullfile(root,'analysis.mlx'));
end
