%%
% Generating Figure 3 in:
% R. Talmon and H.-T. Wu, "Latent common manifold learning with alternating
% diffusion: analysis and applications"
%

clear all;
close all;

%% Create data
n=1000;
l=1;
t = [1:n*l]./n;
h_t = t + sin(2*pi*t);

c1 = exp(i*2*pi*t);
c2 = exp(i*2*pi*h_t);

s1 = [real(c1); imag(c1)];
s2 = [real(c2); imag(c2)];


%% Implement AD
ep = 0.05;

% First view
W1 = pdist2(s1', s1', 'euclidean');
ep1 = ep;

K1 = exp(-W1.^2/ep1^2);
D1 = sum(K1, 1);
A1 = K1 * inv(diag(D1));

[V1 E1] = eig(A1);
[~, I] = sort(abs(diag(E1)), 'descend');
E1 = E1(I,I);
V1 = V1(:,I);

% Second view
W2 = pdist2(s2', s2', 'euclidean');
ep2 = ep;

K2 = exp(-W2.^2/ep2^2);
D2 = sum(K2, 1);
A2 = K2 * inv(diag(D2));

[V2 E2] = eig(A2);
[~, I] = sort(abs(diag(E2)), 'descend');
E2 = E2(I,I);
V2 = V2(:,I);

% Alternating diffusion
K = A2 * A1;
D = sum(K, 1);
A = K * inv(diag(D));

[V E] = eig(A);
[~, I] = sort(abs(diag(E)), 'descend');
E = E(I,I);
V = V(:,I);

Kb = A1 * A2;
Db = sum(Kb, 1);
Ab = Kb * inv(diag(Db));

[Vb Eb] = eig(Ab);
[~, I] = sort(abs(diag(Eb)), 'descend');
Eb = Eb(I,I);
Vb = Vb(:,I);

Vinterp2 = interp1(h_t, V(:,2), t, 'linear', 'extrap'); %Pullback to S1
Vinterp3 = interp1(h_t, V(:,3), t, 'linear', 'extrap'); %Pullback to S1

%% Plot
figure;
xhat= fft(V1(:,2));
bar([1/n:1/n:l/2], abs(xhat(2:n*l/2+1))); 
axis([0 10*l/n -inf inf]);
set(gca,'fontsize',14);
xlabel('Frequency','fontsize',20);
set(gca, 'XTickMode', 'manual');
set(gca, 'XTick', [0:0.001:0.01]);
print('-depsc','./figures/fft_s1.eps');

figure;
xhat= fft(V2(:,2));
bar([1/n:1/n:l/2], abs(xhat(2:n*l/2+1)));
axis([0 10*l/n -inf inf]);
set(gca,'fontsize',14);
xlabel('Frequency','fontsize',20);
set(gca, 'XTickMode', 'manual');
set(gca, 'XTick', [0:0.001:0.01]);
print('-depsc','./figures/fft_s2.eps');

figure;
xhat= fft(Vinterp2);
bar([1/n:1/n:l/2], abs(xhat(2:n*l/2+1))); 
axis([0 10*l/n -inf inf]);
set(gca,'fontsize',14);
xlabel('Frequency','fontsize',20);
set(gca, 'XTickMode', 'manual');
set(gca, 'XTick', [0:0.001:0.01]);
print('-depsc','./figures/fft_AD1.eps');

figure;
xhat= fft(Vb(:,2));
bar([1/n:1/n:l/2], abs(xhat(2:n*l/2+1))); 
axis([0 10*l/n -inf inf]);
set(gca,'fontsize',14);
xlabel('Frequency','fontsize',20);
set(gca, 'XTickMode', 'manual');
set(gca, 'XTick', [0:0.001:0.01]);
print('-depsc','./figures/fft_AD2.eps');
