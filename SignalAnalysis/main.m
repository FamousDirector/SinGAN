clear
clc
close all

%%

scale_num = 1;

% import all the simulated data

output_dir = dir([pwd '/../Output/RandomSamples/PSU_Data_200ms_part2*']);

row_nums = [];
sim_data = [];
for i = 1:length(output_dir)
    disp(['Loading sim data: ' num2str(i) '/' num2str(length(output_dir)) ])
    % get row numbers for real data
    folder_name = output_dir(i).name;
    folder_name = split(folder_name,'_');
    row_nums = [row_nums str2double(folder_name{5}(4:end))];
    
    % load in simulated data
    sim_data = [sim_data; csvread([output_dir(i).folder '/' output_dir(i).name '/cwt/gen_start_scale=' num2str(scale_num) '/signals.csv'])];
    
end

r_sim_data = [];
for i = 1:size(sim_data,1)
    r_sim_data = [r_sim_data; reshape(sim_data(i,:), [],5)];
end

row_nums = row_nums + 2;


%% get original samples

raw_data = csvread([pwd '/../emg_data/old/PSU_Data_200ms_part2.csv']);
real_classes = raw_data(row_nums,1);
real_data = raw_data(row_nums,2:end);

real_data = [raw_data(row_nums, 2:end); raw_data(row_nums+1, 2:end);raw_data(row_nums+2, 2:end)];

%reshape the real data
r_real_data = [];
for i = 1:size(real_data,1)
    r_real_data = [r_real_data; reshape(real_data(i,:), [],5)];
end


%% extract features

% we should optimize threshold on zc and ssc here
real_features = [getmavfeat(r_real_data, 200,200 ), getzcfeat(r_real_data,200,200),...
    getsscfeat(r_real_data,200,200), getwlfeat(r_real_data,200,200)];


sim_features = [getmavfeat(r_sim_data, 200,200 ), getzcfeat(r_sim_data,200,200),...
    getsscfeat(r_sim_data,200,200), getwlfeat(r_sim_data,200,200)];


%% Project Features

[coeff, real_score,~,~,~,mu] = pca(real_features);


num_generated = 100;
sim_classes = [];
for i = 1:length(real_classes)
    for j = 1:num_generated
        sim_classes = [sim_classes real_classes(i)];
    end
end

[~,     sim_score] = pca((sim_features-mu) * coeff);

%% calculate metrics
real_CHi     = evalclusters(real_features, [real_classes; real_classes; real_classes],'CalinskiHarabasz');
sim_CHi      = evalclusters(sim_features,  sim_classes','CalinskiHarabasz');
combined_CHi = evalclusters([real_features;sim_features],  [real_classes; real_classes; real_classes; sim_classes'],'CalinskiHarabasz');

fprintf('CHi - Real: %.2f, Simulated: %.2f, Combined: %.2f\n', real_CHi.CriterionValues, sim_CHi.CriterionValues, combined_CHi.CriterionValues)

real_DBi     = evalclusters(real_features, [real_classes; real_classes; real_classes],'DaviesBouldin');
sim_DBi      = evalclusters(sim_features,  sim_classes','DaviesBouldin');
combined_DBi = evalclusters([real_features;sim_features],  [real_classes; real_classes; real_classes; sim_classes'],'DaviesBouldin');

fprintf('DBi - Real: %.2f, Simulated: %.2f, Combined: %.2f\n', real_DBi.CriterionValues, sim_DBi.CriterionValues, combined_DBi.CriterionValues)

real_Si     = evalclusters(real_features, [real_classes; real_classes; real_classes],'silhouette');
sim_Si      = evalclusters(sim_features,  sim_classes','silhouette');
combined_Si = evalclusters([real_features;sim_features],  [real_classes; real_classes; real_classes; sim_classes'],'silhouette');

fprintf('Si - Real: %.2f, Simulated: %.2f, Combined: %.2f\n', real_Si.CriterionValues, sim_Si.CriterionValues, combined_Si.CriterionValues)

%% produce plot for real data

figure()
hold on 
unique_classes = unique(real_classes);
colors = hsv(size(unique_classes,1));
for i = 1:length(unique_classes)
    class_mask = real_classes == unique_classes(i);
    class_mask = [class_mask class_mask class_mask];
    plot3(real_score(class_mask,1), real_score(class_mask,2), real_score(class_mask,3),...
            '.','MarkerSize',20,'Color',colors(i,:))
end
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
title('Real Data PCA Feature Space')

%% produce plot for simuluated data
figure()
hold on 
unique_classes = unique(real_classes);
colors = hsv(size(unique_classes,1));
for i = 1:length(unique_classes)
    s_class_mask = sim_classes == unique_classes(i);
    plot3(sim_score(s_class_mask,1), sim_score(s_class_mask,2), sim_score(s_class_mask,3),...
            '.','MarkerSize',20,'Color',colors(i,:))
end
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
title('Sim Data PCA Feature Space')

%% Produce plot of combined data

figure()
subplot(1,2,1)
unique_class = unique(real_classes);

p_sim_data = (sim_features-mu) * coeff; % project simulated data w/
                                   % real data PCA projection
colors = hsv(size(unique_classes,1));
hold on
for i = 1:length(unique_classes)
    class_mask = real_classes == unique_classes(i);
    class_mask = [class_mask class_mask class_mask];
    
    s_class_mask = sim_classes == unique_classes(i);
    
    plot3(real_score(class_mask,1), real_score(class_mask,2), real_score(class_mask,3),'o',...
        p_sim_data(s_class_mask,1), p_sim_data(s_class_mask,2), p_sim_data(s_class_mask,3),'+','Color',colors(i,:))
    
end
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')

subplot(1,2,2)

plot3(real_score(:,1), real_score(:,2), real_score(:,3),'+b',...
    p_sim_data(:,1),p_sim_data(:,2),p_sim_data(:,3),'or')
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
sgtitle('Combined PCA Feature Space')

