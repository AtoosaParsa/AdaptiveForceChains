for ii = 1:6
    firstpoint = length(find(stiffparticles1{ii}(:, 3)>0.001));
    first(ii) = length(stiffparticles1{ii})-firstpoint+1;
    lastpoint =  length(find(stiffparticles1{ii}(:, 3)>0.001));
    last = length(stiffparticles1{ii})-lastpoint+1;
    %p(ii, :) = polyfit(stiffparticles{ii}(first(ii):last, 2)*10^(-3), stiffparticles{ii}(first(ii):last, 3), 1);
    p0 = 0.5e-6;
    dist{ii} = stiffparticles1{ii}(first(ii):round(end/3), 2)-stiffparticles1{ii}(first(ii), 2);
    fun = @(x)sseval(x, stiffparticles1{ii}(first(ii):round(end/3), 3), dist{ii}/1000);
    xstiff(ii) = fminsearchbnd(fun, p0, [0], [2e-4]);
end
idd = 1;
%trystiff = 0.11e-5;
plotfit(xstiff(idd), stiffparticles1{idd}(first(idd):round(end/3), 3), dist{idd}/1000)

%%
for ii = 1:6
    firstpoint = length(find(softparticles{ii}(:, 3)>0.01));
    firstsoft(ii) = length(softparticles{ii})-firstpoint+1;
    lastpoint =  length(find(softparticles{ii}(:, 3)>0.05));
    last = length(softparticles{ii})-lastpoint+1;
    p1(ii, :) = polyfit(softparticles{ii}(firstsoft(ii):last, 2)*10^(-3), softparticles{ii}(firstsoft(ii):last, 3), 1);
    p0 = 0.15e-5;
    dist_soft{ii} = softparticles{ii}(firstsoft(ii):end/2, 2)-softparticles{ii}(firstsoft(ii), 2);
    fun1 = @(x)sseval(x, softparticles{ii}(firstsoft(ii):end/2, 3), dist_soft{ii}/1000);
    xsoft(ii) = fminsearch(fun1, p0);
    %xsoft(ii) = fminsearch(@(x) sum((4/3*x*(3*10^(-3))*(dist_soft{ii}*10^(-3)).^(3/2)-softparticles{ii}(firstsoft(ii):end/2, 3)).^2),p0);
end
idd = 6;
plotfit1(xsoft(idd), softparticles{idd}(firstsoft(idd):end/2, 3), dist_soft{idd}/1000)
display(xsoft)

%% Emtpy shell

for ii = 1:7
    firstpoint = length(find(emptyparticles{ii}(:, 3)>0.01));
    firstempty(ii) = length(emptyparticles{ii})-firstpoint+1;
    p1(ii, :) = polyfit(emptyparticles{ii}(firstempty(ii):last, 2)*10^(-3), emptyparticles{ii}(firstempty(ii):last, 3), 1);
    p0 = 0.65e-5;
    dist_empty{ii} = emptyparticles{ii}(firstempty(ii):end, 2)-emptyparticles{ii}(firstempty(ii), 2);
    fun1 = @(x)sseval(x, emptyparticles{ii}(firstempty(ii):end, 3)-0.01, dist_empty{ii}/1000);
    xempty(ii) = fminsearch(fun1, p0);
    %xsoft(ii) = fminsearch(@(x) sum((4/3*x*(3*10^(-3))*(dist_soft{ii}*10^(-3)).^(3/2)-softparticles{ii}(firstsoft(ii):end/2, 3)).^2),p0);
end
idd = 1;
display(xempty)

counter = 1;
for ii = 1:7
emptyparticles_xs{counter} = emptyparticles{ii}(firstempty(ii):end, 2)-emptyparticles{ii}(firstempty(ii), 2);
emptyparticles_ys{counter} = emptyparticles{ii}(firstempty(ii):end, 3);
counter=counter+1;
plot(emptyparticles{ii}(firstempty(ii):end, 2)-emptyparticles{ii}(firstempty(ii), 2), emptyparticles{ii}(firstempty(ii):end, 3), 'color', 'red', 'Linewidth', 2)
hold on
end
emptyparticlestot = averageerror(emptyparticles_xs, emptyparticles_ys);
[l, p] = boundedline(emptyparticlestot(:, 1), emptyparticlestot(:, 2), emptyparticlestot(:, 3)-emptyparticlestot(1, 3), 'alpha');
%plotfit(xempty(idd), emptyparticles{idd}(firstempty(idd):end, 3)-0.01, dist_empty{idd}/1000)
xlim([0, 0.45])
ylim([0, 0.35])
%% Particle-particle

for ii = 1:length(particleparticle)
    firstpoint = length(find(particleparticle{ii}(:, 3)>0.01));
    firstpp(ii) = length(particleparticle{ii})-firstpoint+1;
    lastpoint =  length(find(particleparticle{ii}(:, 3)>0.05));
    last = length(particleparticle{ii})-lastpoint+1;
    p1(ii, :) = polyfit(particleparticle{ii}(firstpp(ii):last, 2)*10^(-3), particleparticle{ii}(firstpp(ii):last, 3), 1);
    p0 = 2e-6;
    dist_pp{ii} = particleparticle{ii}(firstpp(ii):round(end/2), 2)-particleparticle{ii}(firstpp(ii), 2);
    fun2 = @(x)ssevalpp(x, particleparticle{ii}(firstpp(ii):round(end/2), 3), dist_pp{ii}/1000);
    xpp(ii) = fminsearch(fun2, p0);
    %xsoft(ii) = fminsearch(@(x) sum((4/3*x*(3*10^(-3))*(dist_soft{ii}*10^(-3)).^(3/2)-softparticles{ii}(firstsoft(ii):end/2, 3)).^2),p0);
end
idd = 3;
plotfitpp(xpp(idd), particleparticle{idd}(firstpp(idd):round(end/2), 3), dist_pp{idd}/1000)

%% Correct Force Law

function sse = sseval(x,xdata,ydata)
L=0.006;
a = L/2;
V = x;
D=0.012;
Pbar = xdata/(2*a);

sse = sum((ydata - 2*Pbar*V.*(1+log((8*a^2)./(V*Pbar*D)))).^2);
end

function sse = sseval1(x,xdata,ydata)
L=0.006;
a = L/2;
V = x;
D=0.012;
Pbar = xdata/(2*a);

sse = sum((ydata - Pbar*(V+1.1399e-06).*(1+log((8*a^2)./((V+1.1399e-06)*Pbar*D)))).^2);
end

function plotfit(Vfit, xdata, ydata)

L=0.006;
a = L/2;
V = Vfit;
D=0.012;
Pbar = xdata/(2*a);

plot(xdata, ydata, 'Linewidth', 3)
hold on
plot(xdata, 2*Pbar*V.*(1+log((4*a^2)./(V*Pbar*D))), 'Linewidth', 3)
legend('Experiments', 'Fit', 'Location', 'nw')
xlabel('Force [N]')
ylabel('Displacement [mm]')
ylabel('Displacement [m]')
set(gca, 'Fontsize', 15)
set(gca, 'FontWeight','Bold')
end

function plotfit1(Vfit, xdata, ydata)

L=0.006;
a = L/2;
V = Vfit;
D=0.012;
Pbar = xdata/(2*a);

plot(xdata, ydata, 'Linewidth', 3)
hold on
plot(xdata, Pbar*(V+1.1399e-06).*(1+log((8*a^2)./((V+1.1399e-06)*Pbar*D))), 'Linewidth', 3)
legend('Experiments', 'Fit', 'Location', 'nw')
xlabel('Force [N]')
ylabel('Displacement [mm]')
ylabel('Displacement [m]')
set(gca, 'Fontsize', 15)
set(gca, 'FontWeight','Bold')
end

%% Particle particle fits
function sse = ssevalpp(x,xdata,ydata)
L=0.006;
a = L/2;
V = x;
D=0.012;
Pbar = xdata/(2*a);

sse = sum((ydata - Pbar*(V).*(1+log((16*a^2)./((V)*Pbar*D)))).^2);
end

function plotfitpp(Vfit, xdata, ydata)

L=0.006;
a = L/2;
V = Vfit;
D=0.012;
Pbar = xdata/(2*a);

plot(xdata, ydata, 'Linewidth', 3)
hold on
plot(xdata, Pbar*(V).*(1+log((16*a^2)./((V)*Pbar*D))), 'Linewidth', 3)
legend('Experiments', 'Fit', 'Location', 'nw')
xlabel('Force [N]')
ylabel('Displacement [mm]')
ylabel('Displacement [m]')
set(gca, 'Fontsize', 15)
set(gca, 'FontWeight','Bold')
end