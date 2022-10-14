%Written by SeHyoun Ahn, Ben Moll and Andreas Schaab
%NEEDS INPUT FROM huggett_initial.m AND huggett_terminal.m
clear all; close all; clc;

global N I amax amin da s z zz aa dt rho

load huggett_initial.mat %start from equilibrium with low lambda2
g0 = sparse(g);
gg0 = sparse(gg);

load huggett_terminal.mat %terminal condition
g_st = sparse(g); v_st = v; r_st = r;

clear r Delta;

T = 20;
N = 100;
dt = T/N;

f = zeros(N,1);
f1 = zeros(N,1);

data{1} = v_st;
data{2} = gg0;
data{3} = Aswitch;

maxit = 100;
convergence_criterion = 10^(-6);

%initial guess of interest rate sequence
x = r_st*ones(N,1);

%evaluate market clearing condition at initial guess
f = huggett_subroutine(x, data);
disp(['convergence criterion at initial guess = ', num2str(max(abs(f)))]) 

% COMPUTE JACOBIAN AT INITIAL GUESS
disp('computing Jacobian at initial guess')
tic
h = spdiags(sqrt(eps)*max(abs(x(:)), 1), 0, N, N); %Should be N x # of conjectured time series

J = zeros(N,N);
for k = 1:N
    disp(['column of Jacobian = ', num2str(k)])
    x1 = x + h(:,k);
    f1 = huggett_subroutine(x1, data);
    J(:,k) = (f1 - f) / h(k,k);
end
toc
%check that the Jacobian has full rank
disp(['Rank of Jacobian = ', num2str(rank(J))])
J0 = J; %save initial Jacobian for future reference

% FIND EQUILIBRIUM INTEREST RATE USING BROYDEN METHOD (DAMPENED VERSION)
%xi = 0.1*(exp(-0.05*(1:N)) - exp(-0.05*N))'; %dampening factor
xi = 0.1; %dampening factor
for it = 1:maxit 
    dx = -J\f; 

    x = x + xi.*dx; %dampened Broyden updating
    f = huggett_subroutine(x, data);
    Sdist(it) = max(abs(f));
        
    J = J + f * dx'/(dx'*dx);  

    disp(['ITERATION = ', num2str(it)])
    disp(['Convergence criterion = ', num2str(Sdist(it))])
     
    if Sdist(it)<convergence_criterion
        break
    end
end

r_t = x;
time = (1:N)'*dt;

plot(time,r_t)
