function transform=center_only(train)
% Example custom fitter: statistics are captured from the training rows only.
mu=mean(train,1); transform=@(new)new-mu;
end
