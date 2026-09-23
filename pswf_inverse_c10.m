%% pswf_inverse_c10.m
% PSWF-based regularization of a 1D bandlimited inverse problem.
%
% This version is designed to make the PSWF structure more visible:
%   * bandwidth c = 10;
%   * 55 PSWF modes, including the spectral transition near n ~ 4c;
%   * a more complicated synthetic signal;
%   * Gauss-Legendre quadrature for accurate PSWF inner products;
%   * naive inversion, TSVD, and Tikhonov regularization;
%   * Morozov discrepancy principle for practical parameter selection;
%   * oracle parameters ONLY for comparison in this synthetic experiment;
%   * comparison of K_c in the PSWF and Legendre bases.
%
% The PSWF construction uses the same normalized Legendre expansion and
% the same 2*pi*c convention as the supplied one-dimensional PSWF codes.
%
% -------------------------------------------------------------------------
clear; clc; close all;

%% PARAMETERS
c          = 10;
m          = 180;       % Legendre truncation used to construct PSWFs
Nmodes     = 55;        % enough to pass through the n ~ 4c transition
Nq         = 500;       % Gauss-Legendre quadrature points
Nplot      = 1601;      % dense plotting grid
noiseLevel = 5e-4;      % relative weighted L2 noise
tau        = 1.05;      % Morozov safety factor (>1)
rng(11);                % reproducible noise

saveFigures = true;

fprintf('============================================================\n');
fprintf('PSWF INVERSE PROBLEM -- c = %.1f\n',c);
fprintf('Legendre truncation m        : %d\n',m);
fprintf('PSWF modes                   : %d\n',Nmodes);
fprintf('Gauss-Legendre points        : %d\n',Nq);
fprintf('Relative noise level         : %.2e\n',noiseLevel);
fprintf('Morozov safety factor tau    : %.3f\n',tau);
fprintf('Expected spectral mass 4c    : %.1f\n',4*c);
fprintf('============================================================\n\n');

%% GAUSS-LEGENDRE QUADRATURE ON [-1,1]
[xq,w] = gauss_legendre_rule(Nq);
xplot = linspace(-1,1,Nplot).';

wnorm = @(v) sqrt(real(sum(w.*abs(v).^2)));

%% BUILD PSWFS FROM THE DIFFERENTIAL-OPERATOR MATRIX
A = prolatematrix_local(c,m);
A = (A+A')/2;  % remove roundoff asymmetry

[V,D] = eig(A);
[chi,idx] = sort(real(diag(D)),'ascend');
V = V(:,idx);

PhiQ    = normalized_legendre_basis(xq,m);
PhiPlot = normalized_legendre_basis(xplot,m);

PsiQ    = PhiQ*V(:,1:Nmodes);
PsiPlot = PhiPlot*V(:,1:Nmodes);

% Numerical normalization.  Apply the same scaling on quadrature and plot grids.
pswfNorm = sqrt(real(sum(w.*abs(PsiQ).^2,1)));
PsiQ     = PsiQ./pswfNorm;
PsiPlot  = PsiPlot./pswfNorm;

Gram = PsiQ'*(w.*PsiQ);
orthErr = norm(Gram-eye(Nmodes),'fro');

%% SINC-KERNEL OPERATOR
% K_c(x,t) = sin(2*pi*c*(x-t))/(pi*(x-t)),
% with its removable diagonal singularity defined by K_c(x,x)=2c.
Dqq = xq-xq.';
K = sinc_kernel(Dqq,c);

%% PSWF EIGENVALUES OF K_c
KPsi = K*(w.*PsiQ);

lambda = real(sum((w.*PsiQ).*KPsi,1)).';
eigResidual = zeros(Nmodes,1);

for n = 1:Nmodes
    eigResidual(n) = wnorm(KPsi(:,n)-lambda(n)*PsiQ(:,n));
end

% True concentration eigenvalues are positive. Tiny negative values could
% only arise from numerical roundoff.
lambda(lambda < 0 & abs(lambda) < 1e-13) = abs(lambda(lambda < 0 & abs(lambda) < 1e-13));

% Protect only against a zero produced at machine precision.
lambdaSafe = max(lambda,1e-15);

%% MORE STRUCTURED TEST SIGNAL
fTrueQ    = test_signal(xq);
fTruePlot = test_signal(xplot);

%% FORWARD DATA
gClean = K*(w.*fTrueQ);

% Also evaluate clean forward data on the plotting grid.
Dpq = xplot-xq.';
Kplot = sinc_kernel(Dpq,c);
gCleanPlot = Kplot*(w.*fTrueQ);

%% ADD CONTROLLED NOISE
noise = randn(Nq,1);
noise = noiseLevel*wnorm(gClean)*noise/wnorm(noise);
gNoisy = gClean+noise;

deltaAbs = wnorm(noise);
actualNoiseLevel = deltaAbs/wnorm(gClean);
morozovTarget = tau*deltaAbs;

%% DATA COEFFICIENTS IN THE PSWF BASIS
b = PsiQ'*(w.*gNoisy);

%% NAIVE INVERSION
coeffNaive = b./lambdaSafe;
fNaiveQ    = PsiQ*coeffNaive;
fNaivePlot = PsiPlot*coeffNaive;
errNaive   = wnorm(fNaiveQ-fTrueQ)/wnorm(fTrueQ);

%% TSVD / SPECTRAL CUTOFF
tsvdError    = zeros(Nmodes,1);
tsvdResidual = zeros(Nmodes,1);

for k = 1:Nmodes
    coeff = zeros(Nmodes,1);
    coeff(1:k) = b(1:k)./lambdaSafe(1:k);

    fK = PsiQ*coeff;
    tsvdError(k) = wnorm(fK-fTrueQ)/wnorm(fTrueQ);
    tsvdResidual(k) = wnorm(K*(w.*fK)-gNoisy);
end

% Oracle cutoff: only available because this is a synthetic experiment.
[errTSVDOracle,kOracle] = min(tsvdError);

% Practical cutoff from Morozov discrepancy principle.
[~,kDP] = min(abs(tsvdResidual-morozovTarget));
errTSVDDP = tsvdError(kDP);

coeffTSVDDP = zeros(Nmodes,1);
coeffTSVDDP(1:kDP) = b(1:kDP)./lambdaSafe(1:kDP);
fTSVDDPQ    = PsiQ*coeffTSVDDP;
fTSVDDPPlot = PsiPlot*coeffTSVDDP;

%% TIKHONOV REGULARIZATION
%
% f_alpha = argmin_f ||K_c f-g^delta||_2^2 + alpha ||f||_2^2
%
% In the PSWF basis:
% a_n(alpha) = lambda_n/(lambda_n^2+alpha) <g^delta,psi_n>.
%
alphaGrid = logspace(-14,-1,240);

tikError    = zeros(size(alphaGrid));
tikResidual = zeros(size(alphaGrid));
tikSolNorm  = zeros(size(alphaGrid));

for j = 1:length(alphaGrid)
    alpha = alphaGrid(j);

    filt = lambdaSafe./(lambdaSafe.^2+alpha);
    coeff = filt.*b;
    fAlpha = PsiQ*coeff;

    tikError(j) = wnorm(fAlpha-fTrueQ)/wnorm(fTrueQ);
    tikResidual(j) = wnorm(K*(w.*fAlpha)-gNoisy);
    tikSolNorm(j) = wnorm(fAlpha);
end

% Practical alpha: Morozov discrepancy principle.
% Choose alpha so that ||K f_alpha-g^delta|| ~= tau*delta.
[~,jDP] = min(abs(tikResidual-morozovTarget));
alphaDP = alphaGrid(jDP);
errTikDP = tikError(jDP);

% Oracle alpha: for benchmarking only; NOT available with real data.
[errTikOracle,jOracle] = min(tikError);
alphaOracle = alphaGrid(jOracle);

coeffTikDP = (lambdaSafe./(lambdaSafe.^2+alphaDP)).*b;
fTikDPQ    = PsiQ*coeffTikDP;
fTikDPPlot = PsiPlot*coeffTikDP;

%% PICARD QUANTITIES
picardData = abs(b);
picardQuotient = abs(b./lambdaSafe);

%% WHY PSWFS ARE ADAPTED TO THIS OPERATOR
% Compare the representation of K_c in:
%   (i) the PSWF basis,
%   (ii) the normalized Legendre basis.
%
% PSWFs should diagonalize K_c. Legendre polynomials do not.

Mcompare = min(45,Nmodes);

PsiC = PsiQ(:,1:Mcompare);
LegC = PhiQ(:,1:Mcompare);

Apswf = PsiC'*(w.*(K*(w.*PsiC)));
Aleg  = LegC'*(w.*(K*(w.*LegC)));

offdiag = @(B) norm(B-diag(diag(B)),'fro')/norm(B,'fro');

offPSWF = offdiag(Apswf);
offLeg  = offdiag(Aleg);

%% PRINT SUMMARY
fprintf('PSWF orthogonality error ||G-I||_F        : %.3e\n',orthErr);
fprintf('Maximum eigenfunction residual            : %.3e\n',max(eigResidual));
fprintf('Sum of the %d computed eigenvalues         : %.12f\n',Nmodes,sum(lambda));
fprintf('Theoretical total spectral mass 4c        : %.12f\n',4*c);
fprintf('Actual relative noise level               : %.3e\n\n',actualNoiseLevel);

fprintf('RECONSTRUCTION RESULTS\n');
fprintf('------------------------------------------------------------\n');
fprintf('Naive inverse (%d modes) relative error      : %.6e\n',Nmodes,errNaive);
fprintf('TSVD Morozov cutoff k = %d, error           : %.6e\n',kDP,errTSVDDP);
fprintf('TSVD oracle cutoff k = %d, error            : %.6e\n',kOracle,errTSVDOracle);
fprintf('Tikhonov Morozov alpha = %.6e, error       : %.6e\n',alphaDP,errTikDP);
fprintf('Tikhonov oracle alpha  = %.6e, error       : %.6e\n',alphaOracle,errTikOracle);
fprintf('Morozov residual target tau*delta           : %.6e\n',morozovTarget);
fprintf('Residual at selected Tikhonov alpha         : %.6e\n',tikResidual(jDP));
fprintf('------------------------------------------------------------\n\n');

fprintf('OPERATOR REPRESENTATION\n');
fprintf('PSWF relative off-diagonal Frobenius norm    : %.3e\n',offPSWF);
fprintf('Legendre relative off-diagonal Frobenius norm: %.3e\n\n',offLeg);

fprintf('Selected concentration eigenvalues around the transition:\n');
fprintf(' n          lambda_n\n');
for n = max(1,round(4*c)-6):min(Nmodes,round(4*c)+10)
    fprintf('%2d     %.10e\n',n-1,lambda(n));
end

%% FIGURE 1: FORWARD PROBLEM
figure('Color','w','Name','Forward problem c=10');

subplot(2,1,1);
plot(xplot,fTruePlot,'LineWidth',1.8);
grid on;
xlabel('x');
ylabel('f(x)');
title('Structured true signal');

subplot(2,1,2);
plot(xplot,gCleanPlot,'LineWidth',1.6); hold on;
plot(xq,gNoisy,'.','MarkerSize',4);
grid on;
xlabel('x');
ylabel('data');
legend('Clean K_cf','Noisy g^\delta','Location','best');
title(sprintf('Forward data, relative noise %.1e',actualNoiseLevel));

%% FIGURE 2: EIGENVALUE SPECTRUM AND SPECTRAL MASS
figure('Color','w','Name','PSWF eigenvalue spectrum c=10');

subplot(2,1,1);
semilogy(0:Nmodes-1,lambdaSafe,'o-','LineWidth',1.4,'MarkerSize',4);
grid on; hold on;
xline(4*c,'--','4c','LabelVerticalAlignment','bottom');
xlabel('PSWF index n');
ylabel('\lambda_n');
title('Concentration eigenvalue decay');

subplot(2,1,2);
plot(0:Nmodes-1,cumsum(lambda),'o-','LineWidth',1.4,'MarkerSize',4);
grid on; hold on;
yline(4*c,'--','4c');
xlabel('Largest included index n');
ylabel('\Sigma_{j=0}^{n}\lambda_j');
title('Accumulated spectral mass');

%% FIGURE 3: REGULARIZATION DIAGNOSTICS
figure('Color','w','Name','Regularization diagnostics c=10');

subplot(2,1,1);
semilogy(1:Nmodes,tsvdError,'o-','LineWidth',1.3); hold on;
semilogy(kDP,errTSVDDP,'o','MarkerSize',8,'LineWidth',2);
semilogy(kOracle,errTSVDOracle,'s','MarkerSize',8,'LineWidth',2);
grid on;
xlabel('Number of retained modes k');
ylabel('Relative L^2 error');
legend('TSVD error','Morozov cutoff','Oracle cutoff','Location','best');
title('TSVD cutoff study');

subplot(2,1,2);
semilogy(0:Nmodes-1,lambdaSafe,'o-','LineWidth',1.2); hold on;
semilogy(0:Nmodes-1,picardData,'s-','LineWidth',1.1);
semilogy(0:Nmodes-1,picardQuotient,'^-','LineWidth',1.1);
grid on;
xlabel('PSWF index n');
ylabel('Magnitude');
legend('|\lambda_n|','|<g^\delta,\psi_n>|', ...
       '|<g^\delta,\psi_n>/\lambda_n|','Location','best');
title('Picard-type diagnostic');

%% FIGURE 4: HOW ALPHA IS CHOSEN
figure('Color','w','Name','Tikhonov alpha selection c=10');

subplot(2,1,1);
loglog(alphaGrid,tikError,'LineWidth',1.5); hold on;
loglog(alphaDP,errTikDP,'o','MarkerSize',8,'LineWidth',2);
loglog(alphaOracle,errTikOracle,'s','MarkerSize',8,'LineWidth',2);
grid on;
xlabel('\alpha');
ylabel('Relative L^2 error');
legend('Error curve','Morozov \alpha','Oracle \alpha','Location','best');
title('Tikhonov reconstruction error');

subplot(2,1,2);
loglog(alphaGrid,tikResidual,'LineWidth',1.5); hold on;
yline(morozovTarget,'--','\tau\delta');
loglog(alphaDP,tikResidual(jDP),'o','MarkerSize',8,'LineWidth',2);
grid on;
xlabel('\alpha');
ylabel('||K_cf_\alpha^\delta-g^\delta||_2');
legend('Residual','Morozov target','Selected \alpha','Location','best');
title(sprintf('Morozov discrepancy principle, \\tau = %.2f',tau));

%% FIGURE 5: RECONSTRUCTIONS
figure('Color','w','Name','Reconstructions c=10');

subplot(2,2,1);
plot(xplot,fTruePlot,'LineWidth',1.7);
grid on;
xlabel('x'); ylabel('f');
title('True signal');

subplot(2,2,2);
plot(xplot,fNaivePlot,'LineWidth',1.1);
grid on;
xlabel('x'); ylabel('reconstruction');
title(sprintf('Naive inverse, error %.2e',errNaive));

subplot(2,2,3);
plot(xplot,fTruePlot,'LineWidth',1.4); hold on;
plot(xplot,fTSVDDPPlot,'--','LineWidth',1.4);
grid on;
xlabel('x'); ylabel('signal');
legend('True','TSVD','Location','best');
title(sprintf('TSVD (Morozov k=%d), error %.3e',kDP,errTSVDDP));

subplot(2,2,4);
plot(xplot,fTruePlot,'LineWidth',1.4); hold on;
plot(xplot,fTikDPPlot,'--','LineWidth',1.4);
grid on;
xlabel('x'); ylabel('signal');
legend('True','Tikhonov','Location','best');
title(sprintf('Tikhonov (Morozov), error %.3e',errTikDP));

%% FIGURE 6: WHY PSWF BASIS IS NATURAL
figure('Color','w','Name','Operator representation comparison');

subplot(1,2,1);
imagesc(log10(abs(Apswf)+1e-16));
axis image;
colorbar;
xlabel('n'); ylabel('m');
title(sprintf('PSWF basis: log_{10}|A_{mn}|, offdiag %.1e',offPSWF));

subplot(1,2,2);
imagesc(log10(abs(Aleg)+1e-16));
axis image;
colorbar;
xlabel('n'); ylabel('m');
title(sprintf('Legendre basis: log_{10}|A_{mn}|, offdiag %.2f',offLeg));

%% SAVE FIGURES
if saveFigures
    if ~exist('results','dir')
        mkdir('results');
    end

    figNames = { ...
        'basis_comparison_c10.png', ...
        'reconstructions_c10.png', ...
        'tikhonov_alpha_selection_c10.png', ...
        'regularization_diagnostics_c10.png', ...
        'eigenvalue_spectrum_c10.png', ...
        'forward_problem_c10.png'};

    figs = findobj('Type','figure');

    for j = 1:min(length(figs),length(figNames))
        exportgraphics(figs(j),fullfile('results',figNames{j}),'Resolution',180);
    end

    fprintf('\nFigures saved in the results/ directory.\n');
end

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function y = test_signal(x)
% A deliberately more structured signal than the first demonstration.
% It contains three localized components, an oscillatory chirp-like term,
% a near-band-edge sinusoid, and a small linear trend.

y = 0.95*exp(-110*(x+0.58).^2) ...
  - 0.75*exp(-75*(x+0.12).^2) ...
  + 0.90*exp(-95*(x-0.42).^2) ...
  + 0.24*cos(12*pi*x+5*x.^2) ...
  + 0.16*sin(18*pi*x) ...
  + 0.10*x;
end

function K = sinc_kernel(D,c)
% Matrix of sin(2*pi*c*D)/(pi*D), with the removable value 2c at D=0.

K = zeros(size(D));
mask = abs(D) > 100*eps;

K(mask) = sin(2*pi*c*D(mask))./(pi*D(mask));
K(~mask) = 2*c;
end

function [x,w] = gauss_legendre_rule(N)
% N-point Gauss-Legendre rule on [-1,1], via the Golub-Welsch construction.

k = (1:N-1).';
beta = k./sqrt(4*k.^2-1);

J = diag(beta,1)+diag(beta,-1);

[V,D] = eig(J);
[x,idx] = sort(diag(D),'ascend');
V = V(:,idx);

w = 2*(V(1,:).^2).';
end

function Phi = normalized_legendre_basis(x,m)
% Phi(:,k+1) = sqrt(k+1/2) P_k(x), k=0,...,m-1.

N = length(x);
Phi = zeros(N,m);

Pnm1 = ones(N,1);
Phi(:,1) = sqrt(1/2)*Pnm1;

if m == 1
    return;
end

Pn = x;
Phi(:,2) = sqrt(3/2)*Pn;

for n = 1:m-2
    Pnp1 = ((2*n+1).*x.*Pn-n.*Pnm1)/(n+1);
    Phi(:,n+2) = sqrt(n+3/2)*Pnp1;

    Pnm1 = Pn;
    Pn = Pnp1;
end
end

function M = prolatematrix_local(c,m)
% Matrix used in the supplied one-dimensional PSWF construction.
% It corresponds to the 2*pi*c normalization.

M = zeros(m,m);

for i = 2:m-1
    M(i-1,i+1) = (4*pi^2*c^2)*i*(i-1) / ...
        ((2*i-1)*sqrt((2*i+1)*(2*i-3)));
end

for i = 1:m
    M(i,i) = (i-1)*i + ...
        ((4*pi^2*c^2)*(2*i^2-2*i-1)) / ...
        ((2*i+1)*(2*i-3));
end

for i = 3:m
    M(i,i-2) = (4*pi^2*c^2)*(i-1)*(i-2) / ...
        ((2*i-3)*sqrt((2*i-5)*(2*i-1)));
end
end
