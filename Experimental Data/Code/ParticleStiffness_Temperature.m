%%
load allstiffdata.mat
idx_group = {[1; 2];
             [3; 4; 5];
             [6; 7; 8];
             [9; 10; 11];
             [12; 13; 14];
             [15; 16];
             [17; 18; 19]}; % tests at temperatures [27, 35.8, 42.29, 52.63, 61.84, 66.53, 75.68]

fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',0,...
               'Upper',10,...
               'StartPoint',1);
ft = fittype('a*x^1.5','options',fo);

K_all = nan(19, 1);
F_disp_all{19, 1} = [];

for ii = 1:19
    F_disp = allstiffdata{ii};
    idx_start = find(F_disp(:, 3) > 0.01, 1);
    F = F_disp(idx_start:end, 3);
    disp = (F_disp(idx_start:end, 2) - F_disp(idx_start - 1, 2)) / 2;
    idx = (F > 0.01) & (F < 1.51);
%     if ii < 14.5
%         idx = (F > 0.01) & (F < 2);
%     else
%         idx = (F > 0.01) & (disp < 2);
%     end
    % idx = (F_disp(:, 3) > 0.01);
    disp_all = disp(idx);
    F_all = F(idx) - 0.01;
    [curve, gof] = fit(disp_all, F_all, ft);
    K_all(ii) = curve.a * sqrt(1.5) / 4 * 1e6; % in pascal
end