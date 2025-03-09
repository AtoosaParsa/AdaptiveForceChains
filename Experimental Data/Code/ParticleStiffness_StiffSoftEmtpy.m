%% example to fit stiffness of the soft particles
load softparticles.mat
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',0,...
               'Upper',10,...
               'StartPoint',1);
ft = fittype('a*x^1.5','options',fo);

stiffness = zeros(7, 1);

for ii = 1:7
    F_disp = softparticles{ii};
    idx_start = find(F_disp(:, 3) > 0.01, 1);
    idx = (F_disp(:, 3) > 0.01) & (F_disp(:, 3) < 1.5);
    % idx = (F_disp(:, 3) > 0.01);
    disp_all = F_disp(idx, 2);
    disp_all = (disp_all - F_disp(idx_start - 1, 2)) / 2;
    F_all = F_disp(idx, 3) - 0.01;
    [curve, gof] = fit(disp_all, F_all, ft);
    stiffness(ii) = curve.a * sqrt(1.5) / 4 * 1e6; % in Pascal
end