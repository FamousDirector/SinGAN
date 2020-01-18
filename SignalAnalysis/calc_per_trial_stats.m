clear
clc
close all

%% load real raw data

raw_data = csvread('/home/jamesc/PycharmProjects/SinGAN/data/reb_emg_data.csv');

%% parameters
scale_num = 0;

num_generated = 1000;
num_of_channels = 8;
num_of_classes = 7;
num_of_trials  = 4;
num_of_subjects  = 5;
num_of_windows = 56;
chosen_window = 28;
start_offset = 0;

count = start_offset;


unique_classes = [1:num_of_classes]';

for i = 1:(num_of_subjects)
    for j = 1:(num_of_trials)
    
        % prepare data

        training_row = (i-1)*num_of_classes*num_of_windows*num_of_trials +...
                    (j-1)*num_of_windows...
                    + start_offset + chosen_window;

        train_data = [];
        real_data = [];
        sim_data = [];

        train_classes = [];
        real_classes = [];
        sim_classes = [];

        for cl = 1:num_of_classes

            r = (cl-1)*num_of_windows*num_of_trials + start_offset + training_row;
            train_data = [train_data; reshape(raw_data(r,4:end),[],num_of_channels)];
%             train_data = [train_data; reshape(raw_data(r,[4:203 403:602 802:1001 1201:1400]),[],num_of_channels)];
            train_classes = [train_classes; cl];

            tmp = csvread(['/home/jamesc/PycharmProjects/SinGAN/Output/RandomSamples/reb_emg_data_row' num2str(r-1) '/stft/gen_start_scale=' num2str(scale_num) '/signals.csv']);
            sim_data = [sim_data; reshape(tmp(1:num_generated, :),[],num_of_channels)];

            for a = 1:num_generated
                sim_classes = [sim_classes cl];
            end

            for xr = 1:num_of_windows
                if chosen_window ~= xr
                    dif = xr - chosen_window;
                    real_data = [real_data; reshape(raw_data(r + dif,4:end),[],num_of_channels)];
%                     real_data = [real_data; reshape(raw_data(r + dif,[4:203 403:602 802:1001 1201:1400]),[],num_of_channels)];
                    real_classes = [real_classes; raw_data(r + dif,2)];
                end            
            end               
        end


        % create class labels

        % extract features

        train_features = [getmavfeat(train_data, 200,200), getzcfeat(train_data,200,200),...
        getsscfeat(train_data,200,200), getwlfeat(train_data,200,200)];

        real_features = [getmavfeat(real_data, 200,200 ), getzcfeat(real_data,200,200),...
        getsscfeat(real_data,200,200), getwlfeat(real_data,200,200)];

        sim_features = [getmavfeat(sim_data, 200,200 ), getzcfeat(sim_data,200,200),...
        getsscfeat(sim_data,200,200), getwlfeat(sim_data,200,200)];

        % calculate metrics
        real_CHi     = evalclusters(real_features, real_classes,'CalinskiHarabasz');
        sim_CHi      = evalclusters(sim_features,  sim_classes','CalinskiHarabasz');
        combined_CHi = evalclusters([real_features;sim_features],  [real_classes; sim_classes'],'CalinskiHarabasz');


        real_DBi     = evalclusters(real_features, [real_classes],'DaviesBouldin');
        sim_DBi      = evalclusters(sim_features,  sim_classes','DaviesBouldin');
        combined_DBi = evalclusters([real_features;sim_features],  [real_classes; sim_classes'],'DaviesBouldin');


        real_Si     = evalclusters(real_features, [real_classes],'silhouette');
        sim_Si      = evalclusters(sim_features,  sim_classes','silhouette');
        combined_Si = evalclusters([real_features;sim_features],  [real_classes; sim_classes'],'silhouette');


        % display metrics
        fprintf('Subject %.i, Trial %.i\n', i, j)
        fprintf('%.2f\n%.2f\n%.2f\n', real_CHi.CriterionValues, sim_CHi.CriterionValues, combined_CHi.CriterionValues)
        fprintf('%.2f\n%.2f\n%.2f\n', real_DBi.CriterionValues, sim_DBi.CriterionValues, combined_DBi.CriterionValues)
        fprintf('%.2f\n%.2f\n%.2f\n', real_Si.CriterionValues, sim_Si.CriterionValues, combined_Si.CriterionValues)

%         % Project Features
% 
%         [coeff, real_score,~,~,~,mu] = pca(real_features);
%         [~, train_score] = pca((train_features-mu) * coeff);
%         [~, sim_score] = pca((sim_features-mu) * coeff);
% 
%         % calculatet axis limits
%         
%         x_min = min([min(train_score(:,1)),...
%                 min(real_score(:,1)),...
%                 min(sim_score(:,1))]);
%         x_max = max([max(train_score(:,1)),...
%                 max(real_score(:,1)),...
%                 max(sim_score(:,1))]);
% 
%         y_min = min([min(train_score(:,2)),...
%                 min(real_score(:,2)),...
%                 min(sim_score(:,2))]);
%         y_max = max([max(train_score(:,2)),...
%                 max(real_score(:,2)),...
%                 max(sim_score(:,2))]);
% 
%         z_min = min([min(train_score(:,3)),...
%                 min(real_score(:,3)),...
%                 min(sim_score(:,3))]);
%         z_max = max([max(train_score(:,3)),...
%                 max(real_score(:,3)),...
%                 max(sim_score(:,3))]);
% 
%         % produce cluster plot for training data
%             
%         figure()
%         hold on     
%         colors = hsv(size(unique_classes,1));
%         for i = 1:length(unique_classes)
%             class_mask = train_classes == unique_classes(i);
%             plot3(train_score(class_mask,1), train_score(class_mask,2), train_score(class_mask,3),...
%                     '.','MarkerSize',25,'Color',colors(i,:))
%         end
%         xlabel('PC1')
%         ylabel('PC2')
%         zlabel('PC3')
%         view(3)
%         grid on
%         xlim([x_min, x_max])
%         ylim([y_min, y_max])
%         zlim([z_min, z_max])
% 
%         % produce cluster plot for real data
% 
%         figure()
%         hold on 
%         for i = 1:length(unique_classes)
%             class_mask = real_classes == unique_classes(i);
%             plot3(real_score(class_mask,1), real_score(class_mask,2), real_score(class_mask,3),...
%                     '.','MarkerSize',10,'Color',colors(i,:))
%         end
%         xlabel('PC1')
%         ylabel('PC2')
%         zlabel('PC3')
%         view(3)
%         grid on
%         xlim([x_min, x_max])
%         ylim([y_min, y_max])
%         zlim([z_min, z_max])
% 
%         % produce plot cluster for simuluated data
%         figure()
%         hold on 
%         for i = 1:length(unique_classes)
%             s_class_mask = sim_classes == unique_classes(i);
%             plot3(sim_score(s_class_mask,1), sim_score(s_class_mask,2), sim_score(s_class_mask,3),...
%                     '.','MarkerSize',10,'Color',colors(i,:))
%         end
%         xlabel('PC1')
%         ylabel('PC2')
%         zlabel('PC3')
%         view(3)
%         grid on
%         xlim([x_min, x_max])
%         ylim([y_min, y_max])
%         zlim([z_min, z_max])
        
    end
end    



