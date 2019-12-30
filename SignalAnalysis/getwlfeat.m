%
% GETWLFEAT Gets the Waveform Length feature.
%
% feat = getwlfeat(x,winsize,wininc,[])
%
% This function computes the waveform length feature of the signals in x,
% which are stored in columns.
%
% The signals in x are divided into multiple windows of size
% winsize and the windows are space wininc apart.
%
% Inputs
%    x: 		columns of signals
%    winsize:	window size
%    wininc:	spacing of the windows (winsize)
%    datawin:   window for data (e.g. Hamming, default rectangular)
%               must have dimensions of (winsize,1)
%
% Outputs
%    feat:     waveform length value in a 2 dimensional matrix
%              dim1 window
%              dim2 feature (col i is the features for the signal in column i of x)

function feat = getwlfeat(x,winsize,wininc,option) %#ok<INUSD>

if nargin < 3
   if nargin < 2
      winsize = size(x,1);
   end
   wininc = winsize;
end

datawin = ones(winsize,1);

datasize = size(x,1);
Nsignals = size(x,2);
numwin = floor((datasize - winsize)/wininc)+1;

feat = zeros(numwin,Nsignals);

st = 1;
en = winsize;

for i = 1:numwin
   curwin = x(st:en,:).*repmat(datawin,1,Nsignals);
   
   % WL
   feat(i,:) = sum(abs(diff(curwin)));
   
   st = st + wininc;
   en = en + wininc;
end
