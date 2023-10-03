% import data
clear all;
data=importdata('data/1-weld_depression_data_train.csv');
data_test=importdata('data/1-weld_depression_data_test.csv');

% X and Y
X=data(:,1);
Y=data(:,2);

% initialize 
theta_matrix=[];
for j=1:10

theta_0=[-0.2-rand(1)*0.1;190+rand(1)*10;0.1];
theta_old=theta_0;

% star iteration
max_step=100;
eta=0.001;
F=[];
tolerance=0.0001;
step=0;
dis=100;

while step <= max_step && dis >= tolerance

    w_0=theta_old(1,1);
    lamta=theta_old(2,1);
    sigma=theta_old(3,1);
    f_w=w_0.*exp(-pi.*abs(X)./lamta).*(cos(pi.*abs(X)./lamta)+sin(pi.*abs(X)./lamta));
    e=(f_w-Y)./sigma;
    F_1=e.*exp(-pi.*abs(X)./lamta).*(cos(pi.*abs(X)./lamta)+sin(pi.*abs(X)./lamta));
    F_2=2.*e.*exp(-pi.*abs(X)./lamta).*w_0.*pi.*abs(X)./(lamta.*lamta);
    F_3=e.*e./(sigma^3);

    F(1,1)=sum(F_1);
    F(2,1)=sum(F_2);
    F(3,1)=(length(X)/sigma)-0.5*sum(F_3);

    theta_new=theta_old-eta.*F;

    dis=norm(theta_new-theta_old);
    theta_old=theta_new;
    step=step+1;
end

theta_matrix=[theta_matrix,theta_new];
end

% store result
w_0=mean(theta_matrix(1,:))
lamta=mean(theta_matrix(2,:))

%% evaluate training data
% predict Y using learned model for training data
f_w = w_0.*exp(-pi.*abs(X)./lamta).*(cos(pi.*abs(X)./lamta)+sin(pi.*abs(X)./lamta));

% calculate MSE
MSE_train = sum((f_w-Y).*(f_w-Y))/length(X)

% calculate R_square
R_square_train = 1-sum((f_w-Y).*(f_w-Y))./sum((Y-mean(Y)).*(Y-mean(Y)))

% calculate Pearson’s Correlation
P_corr = corrcoef(Y,f_w);
P_corr_train = P_corr(1,2)

% calculate Kendall’s Tau
Ken_tau_train = corr(Y,f_w,'Type','Kendall')



%% evaluate testing data 
X_test=data_test(:,1);
Y_test=data_test(:,2);
Y_pred=w_0.*exp(-pi.*abs(X_test)./lamta).*(cos(pi.*abs(X_test)./lamta)+sin(pi.*abs(X_test)./lamta));

% calculate MSE
MSE_test = sum((Y_pred-Y_test).*(Y_pred-Y_test))/length(X_test)

% calculate R_square
R_square_test = 1-sum((Y_pred-Y_test).*(Y_pred-Y_test))./sum((Y_test-mean(Y_test)).*(Y_test-mean(Y_test)))

% calculate Pearson’s Correlation
P_corr = corrcoef(Y_test,Y_pred);
P_corr_test = P_corr(1,2)

% calculate Kendall’s Tau
Ken_tau_test = corr(Y_test,Y_pred,'Type','Kendall')

%% Plot data
% plot figure
figure;
scatter(X,Y);
hold on;
scatter(X_test,Y_test)
hold on;

X_fit=-238:1:238;
f_w=w_0.*exp(-pi.*abs(X_fit)./lamta).*(cos(pi.*abs(X_fit)./lamta)+sin(pi.*abs(X_fit)./lamta));

plot(X_fit,f_w,'LineWidth',5);
legend('Train data','Test data', 'Model-ML');

xlabel('Longitudinal Position d [mm]');
ylabel('Weld Depression w(d) [mm]');
ylim([-0.5 0.2]);
saveas(gca,'1_Result_ML.jpg');
