function transform = fit_representation(train,spec)
% Fit on training rows only. Custom spec: @(train) @(new) transformedRows.
% Built-ins: identity, standardize, pca, pipeline, autoencoder.
if nargin<2 || isempty(spec), spec=struct('kind','identity'); end
if isa(spec,'function_handle')
    raw=spec(train);
else
    kind=lower(string(spec.kind));
    switch kind
        case 'identity'
            raw=@(x)x;
        case 'standardize'
            mu=mean(train,1); scale=max(std(train,1,1),1e-8);
            raw=@(x)(x-mu)./scale;
        case 'pca'
            if ~isfield(spec,'components'), spec.components=2; end
            if ~isfield(spec,'standardize'), spec.standardize=true; end
            if ~isfield(spec,'whiten'), spec.whiten=false; end
            if spec.standardize, norm=fit_representation(train,struct('kind','standardize')); else, norm=@(x)x; end
            z=norm(train); mu=mean(z,1);
            assert(size(z,1)>=2 && any(var(z,1,1)>0),'audit:PCA','PCA needs varying training data.');
            [~,s,v]=svd(z-mu,'econ'); variance=diag(s).^2/(size(z,1)-1);
            k=spec.components;
            assert(isscalar(k)&&isfinite(k)&&k>0,'audit:PCA','Invalid component count/fraction.');
            if k<1, k=find(cumsum(variance)/sum(variance)>=k,1); end
            assert(k==fix(k)&&k<=min(size(z)),'audit:PCA','Components exceed training dimensions.');
            v=v(:,1:k); scale=ones(1,k);
            if spec.whiten
                assert(all(variance(1:k)>eps),'audit:PCA','Cannot whiten zero variance.');
                scale=sqrt(variance(1:k))';
            end
            raw=@(x)((norm(x)-mu)*v)./scale;
        case 'pipeline'
            transforms=cell(size(spec.steps)); z=train;
            for i=1:numel(spec.steps)
                transforms{i}=fit_representation(z,spec.steps{i}); z=transforms{i}(z);
            end
            raw=@(x)apply_pipeline(x,transforms);
        case 'autoencoder'
            % Optional Deep Learning Toolbox; native trainAutoencoder backend.
            assert(exist('trainAutoencoder','file')==2,'audit:Toolbox','Autoencoder requires Deep Learning Toolbox.');
            if ~isfield(spec,'components'), spec.components=2; end
            if ~isfield(spec,'epochs'), spec.epochs=50; end
            if ~isfield(spec,'seed'), spec.seed=42; end
            assert(size(train,1)>=5,'audit:Autoencoder','At least five training rows are required.');
            state=rng; cleanup=onCleanup(@()rng(state)); rng(spec.seed,'twister');
            norm=fit_representation(train,struct('kind','standardize'));
            model=trainAutoencoder(norm(train)',spec.components,'MaxEpochs',spec.epochs, ...
                'ScaleData',false,'ShowProgressWindow',false,'UseGPU',false);
            raw=@(x)encode(model,norm(x)')';
        otherwise
            error('audit:Representation','Unknown representation: %s',kind);
    end
end
assert(isa(raw,'function_handle'),'audit:Representation','A fitter must return a function handle.');
probe=raw(train); p=size(train,2); w=size(probe,2);
assert(ismatrix(probe)&&size(probe,1)==size(train,1)&&w>0&&all(isfinite(probe),'all'),'audit:Representation','Invalid training transform.');
transform=@(x)checked(raw,x,p,w);
end
function z=checked(raw,x,p,w)
assert(isnumeric(x)&&isreal(x)&&size(x,2)==p&&all(isfinite(x),'all'),'audit:Representation','Invalid transform input.');
z=raw(x);
assert(isnumeric(z)&&isreal(z)&&isequal(size(z),[size(x,1) w])&&all(isfinite(z),'all'),'audit:Representation','Transform must preserve rows and width.');
end
function z=apply_pipeline(z,transforms)
for i=1:numel(transforms), z=transforms{i}(z); end
end
