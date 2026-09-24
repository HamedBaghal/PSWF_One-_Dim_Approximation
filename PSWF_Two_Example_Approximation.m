

clear; clc; close all;
c  = 10;
m  = 240;
N  = 44;
Nq = 700;
Nx = 2001;

%% Gauss-Legendre nodes and plotting grid
[xq,wq] = gauss_legendre_rule(Nq);
x = linspace(-1,1,Nx).';

%% Construct the first N PSWFs
A = prolatematrix(c,m);
A = (A + A')/2;

[V,D] = eig(A);
[~,idx] = sort(real(diag(D)));
V = V(:,idx);

PhiQ = normalized_legendre_basis(xq,m);
PhiX = normalized_legendre_basis(x,m);

PsiQ = PhiQ*V(:,1:N);
PsiX = PhiX*V(:,1:N);

% Normalize the PSWFs in L^2[-1,1]
pswfNorm = sqrt(real(sum(wq.*abs(PsiQ).^2,1)));
PsiQ = PsiQ./pswfNorm;
PsiX = PsiX./pswfNorm;

%% Example 1: exactly bandlimited signal
f1q = exact_bandlimited_signal(xq);
f1  = exact_bandlimited_signal(x);

% PSWF coefficients:
% a_n = integral_{-1}^1 f(x) psi_n(x) dx
% computed by Gauss-Legendre quadrature
a1 = PsiQ'*(wq.*f1q);

f1Nq = PsiQ*a1;
f1N  = PsiX*a1;

L2err1 = sqrt(real(sum(wq.*abs(f1q-f1Nq).^2)));
L2norm1 = sqrt(real(sum(wq.*abs(f1q).^2)));
relerr1 = L2err1/L2norm1;

figure('Color','w');
plot(x,f1,'LineWidth',1.5); hold on;
plot(x,f1N,'--','LineWidth',1.5);
grid on;
xlabel('x');
ylabel('f(x)');
legend('Original signal','PSWF approximation','Location','best');
title(sprintf('Exactly bandlimited example: c=%g, N=%d',c,N));
exportgraphics(gcf,'exact_bandlimited_example.png','Resolution',180);

%% Example 2: almost-bandlimited signal
f2q = almost_bandlimited_signal(xq);
f2  = almost_bandlimited_signal(x);

a2 = PsiQ'*(wq.*f2q);

f2Nq = PsiQ*a2;
f2N  = PsiX*a2;

L2err2 = sqrt(real(sum(wq.*abs(f2q-f2Nq).^2)));
L2norm2 = sqrt(real(sum(wq.*abs(f2q).^2)));
relerr2 = L2err2/L2norm2;

figure('Color','w');
plot(x,f2,'LineWidth',1.5); hold on;
plot(x,f2N,'--','LineWidth',1.5);
grid on;
xlabel('x');
ylabel('f(x)');
legend('Original signal','PSWF approximation','Location','best');
title(sprintf('Almost-bandlimited example: c=%g, N=%d',c,N));
exportgraphics(gcf,'almost_bandlimited_example.png','Resolution',180);

%% Print the errors
fprintf('\nPSWF APPROXIMATION WITH c = %g AND N = %d\n',c,N);
fprintf('--------------------------------------------------\n');
fprintf('Exactly bandlimited signal:\n');
fprintf('  L2 error          = %.6e\n',L2err1);
fprintf('  Relative L2 error = %.6e\n\n',relerr1);

fprintf('Almost-bandlimited signal:\n');
fprintf('  L2 error          = %.6e\n',L2err2);
fprintf('  Relative L2 error = %.6e\n',relerr2);

%% Write errors for the LaTeX file
fid = fopen('example_errors.tex','w');
fprintf(fid,'\\newcommand{\\ExactLTwoError}{%.6e}\n',L2err1);
fprintf(fid,'\\newcommand{\\ExactRelativeError}{%.6e}\n',relerr1);
fprintf(fid,'\\newcommand{\\AlmostLTwoError}{%.6e}\n',L2err2);
fprintf(fid,'\\newcommand{\\AlmostRelativeError}{%.6e}\n',relerr2);
fclose(fid);

%% ------------------------------------------------------------------------
% Local functions
% -------------------------------------------------------------------------

function f = exact_bandlimited_signal(x)

B   = [0.35 0.45 0.55 0.40 0.60 0.50 0.65 0.45 0.50 0.55 0.40 0.35];
xi  = [0.40 1.20 2.00 2.80 3.60 4.40 5.20 6.00 6.80 7.60 8.40 9.20];
tau = [-0.82 -0.66 -0.50 -0.34 -0.18 -0.03 0.11 0.27 0.43 0.58 0.72 0.84];
amp = [0.85 -0.62 0.74 0.55 -0.68 0.80 -0.51 0.66 0.49 -0.58 0.44 0.38];
phi = [0.15 1.10 -0.70 2.00 -1.20 0.55 1.75 -2.20 0.95 2.35 -0.35 1.40];

f = zeros(size(x));

for j = 1:length(B)
    f = f + amp(j).*nsinc(2*B(j).*(x-tau(j))) ...
        .*cos(2*pi*xi(j).*x + phi(j));
end
end

function f = almost_bandlimited_signal(x)

j = (1:20).';

rho = 10;
a = 0.40*(-1).^(j+1)./sqrt(j);
nu = linspace(0.45,9.60,20).';
phi = mod(sqrt(2)*j.^2,2*pi)-pi;

osc = zeros(size(x));
for k = 1:20
    osc = osc + a(k)*cos(2*pi*nu(k).*x + phi(k));
end
osc = exp(-rho*x.^2).*osc;

b = [0.75 -0.58 0.66 -0.47 0.52];
gamma = [12 18 25 35 50];
tau = [-0.70 -0.32 0.05 0.39 0.72];

bumps = zeros(size(x));
for k = 1:5
    bumps = bumps + b(k)*exp(-gamma(k)*(x-tau(k)).^2);
end

f = osc + bumps;
end

function y = nsinc(z)
y = ones(size(z));
mask = abs(z) > 1e-14;
y(mask) = sin(pi*z(mask))./(pi*z(mask));
end

function Phi = normalized_legendre_basis(x,m)

Phi = zeros(length(x),m);

P0 = ones(size(x));
Phi(:,1) = sqrt(1/2)*P0;

if m == 1
    return;
end

P1 = x;
Phi(:,2) = sqrt(3/2)*P1;

Pm1 = P0;
P = P1;

for n = 1:m-2
    Pp1 = ((2*n+1).*x.*P - n.*Pm1)/(n+1);
    Phi(:,n+2) = sqrt(n+3/2)*Pp1;
    Pm1 = P;
    P = Pp1;
end
end

function [x,w] = gauss_legendre_rule(N)

k = (1:N-1).';
beta = k./sqrt(4*k.^2-1);

J = diag(beta,1) + diag(beta,-1);

[V,D] = eig(J);
[x,idx] = sort(diag(D));
V = V(:,idx);

w = 2*(V(1,:).^2).';
end
