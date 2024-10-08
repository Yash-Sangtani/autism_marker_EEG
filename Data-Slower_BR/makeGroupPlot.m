function makeGroupPlot()

% makeGroupPlot() creates group transition plots for controls (stratus) and
% participants with autism (cumulus).
%
% Input files: In the RLSData/ folder, each file loads the variable 
% `transitions`, which contains a cell array of RLS traces corresponding to 
% all of a participant's transitions. Each cell of the cell array contains 
% a specific type of transition as follows:
% 1. rivalry trials left eye to right eye
% 2. rivalry trials right eye to left eye
% 3. rivalry simulation (control trials) 5.7 Hz (low) to 8.5 Hz (high)
% 4. rivalry simulation (control trials) 8.5 Hz (high) to 5.7 Hz (low)

clearvars

% set variables
transHalf = 5;

% data directory
cd RLSData

for iGroup = 1:2 % 2 groups: stratus (controls) and cumulus (autism)
    
    % allocate space to store transitions
    l2hRival.left = []; l2hRival.right = [];
    h2lRival.left = []; h2lRival.right = [];
    l2hSim.f1 = []; l2hSim.f2 = [];
    h2lSim.f1 = []; h2lSim.f2 = [];
    dartSim.f1 = []; dartSim.f2 = [];
    
    groupTLists = {l2hRival h2lRival l2hSim h2lSim dartSim};
    
    % iterate through participants
    if iGroup == 1
        groupLabel = 'stratus';
        groupPlotLabel = 'controls';
    elseif iGroup == 2
        groupLabel = 'cumulus';
        groupPlotLabel = 'autism';
    end
    files = dir([groupLabel '*.mat']);
    
    for iPar = 1:length(files)
        
        % load participant's transitions
        load(files(iPar).name);
        
        %% Collect transitions
        simTrace_s2d = []; % store suppressed to dominant sim traces
        simTrace_d2s = []; % store dominant to suppressed sim traces
        
        % Four types of transitions:
        % 1. rivalry left to right
        % 2. rivalry right to left
        % 3. sim 5.7 Hz (low) to 8.5 Hz (high)
        % 4. sim 8.5 Hz (high) to 5.7 Hz (low)
        for tType = 1:4
            
            % get mean trace of participant's rivalry transitions
            if tType <= 2
                leftMean = nanmean(transitions{tType}.left,1);
                rightMean = nanmean(transitions{tType}.right,1);
                if isnan(leftMean(1))
                    leftMean = [];
                end
                if isnan(rightMean(1))
                    rightMean = [];
                end
                
                % store mean transitions
                groupTLists{tType}.left = [groupTLists{tType}.left; leftMean];
                groupTLists{tType}.right = [groupTLists{tType}.right; rightMean];
                
                % get mean trace of participant's sim transitions
            else
                f1mean = nanmean(transitions{tType}.f1,1);
                f2mean = nanmean(transitions{tType}.f2,1);
                if isnan(f1mean(1))
                    f1mean = [];
                end
                if isnan(f2mean(1))
                    f2mean = [];
                end
                
                % store mean transitions
                groupTLists{tType}.f1 = [groupTLists{tType}.f1; f1mean];
                groupTLists{tType}.f2 = [groupTLists{tType}.f2; f2mean];
                
                % for sim trials, combine low to high and high to low
                % transitions to create one dominant to suppressed trace
                % and one suppressed to dominant trace
                if tType == 3
                    % supp to dom trace
                    simTrace_s2d = [simTrace_s2d; f2mean];
                    % dom to supp trace
                    simTrace_d2s = [simTrace_d2s; f1mean];
                elseif tType == 4
                    % supp to dom trace
                    simTrace_s2d = [simTrace_s2d; f1mean];
                    % dom to supp trace
                    simTrace_d2s = [simTrace_d2s; f2mean];
                end
            end
        end
        
        % store combined sim transitions
        if ~isnan(simTrace_s2d)
            groupTLists{5}.f1 = [groupTLists{5}.f1; nanmean(simTrace_s2d,1)];
        end
        if ~isnan(simTrace_d2s)
            groupTLists{5}.f2 = [groupTLists{5}.f2; nanmean(simTrace_d2s,1)];
        end
    end
    
    %% Rivalry: zscore, combine, and average transitions
    
    zscoredListLeft = [];
    zscoredListRight = [];
    
    % zscore transitions
    zscored_left_l2r = zscore(groupTLists{1}.left,0,2);
    zscored_right_l2r = zscore(groupTLists{1}.right,0,2);
    zscored_left_r2l = zscore(groupTLists{2}.left,0,2);
    zscored_right_r2l = zscore(groupTLists{2}.right,0,2);
    
    % for rivalry trials, combine left to right (flipped) and right to left
    % transitions to create one left eye trace and one right eye trace 
    for iPar = 1:size(zscored_left_l2r,1) % loop through participants
        zscoredListLeft(iPar,:) = nanmean([fliplr(zscored_left_l2r(iPar,:)) ; zscored_left_r2l(iPar,:)] ,1 );
        zscoredListRight(iPar,:) = nanmean([fliplr(zscored_right_l2r(iPar,:)) ; zscored_right_r2l(iPar,:)] ,1 );
    end
    
    % averages across participants
    meanTraceLeft = nanmean(zscoredListLeft,1);
    errorTraceLeft = ste(zscoredListLeft);
    
    meanTraceRight = nanmean(zscoredListRight,1);
    errorTraceRight = ste(zscoredListRight);
    
    
    %% Simulation: zscore and average transitions (already combined above)
    
    zscoredListf1 = zscore(groupTLists{5}.f1,0,2);
    zscoredListf2 = zscore(groupTLists{5}.f2,0,2);
    
    meanTracef1 = nanmean(zscoredListf1,1);
    errorTracef1 = ste(zscoredListf1);
    
    meanTracef2 = nanmean(zscoredListf2,1);
    errorTracef2 = ste(zscoredListf2);
    
    
    %% Plotting
    
    % Plot all rivalry participant traces
    
    figure    
    title([groupPlotLabel ' rivalry: all participants'], 'FontSize', 16)
    hold on
       
    plot([(-transHalf):(1/512):transHalf],zscoredListLeft', 'r','LineWidth',0.25)     
    plot([(-transHalf):(1/512):transHalf],zscoredListRight', 'b','LineWidth',0.25)
    
    ylim([-3 3])
    %vline(0, 'k', 'button press')
    xlabel('Time from button press (s)');
    ylabel('Amplitude');
    legend('Left Eye Red', 'Right Eye Blue');
        
    % Plot all sim participant traces 
    figure    
    title([groupPlotLabel ' simulation: all participants'], 'FontSize', 16)
    hold on
    
    plot([(-transHalf):(1/512):transHalf],zscoredListf1', 'r','LineWidth',0.25)       
    plot([(-transHalf):(1/512):transHalf],zscoredListf2', 'b','LineWidth',0.25)

    ylim([-3 3]);
    %vline(0, 'k', 'button press')   
    xlabel('Time from button press (s)');
    ylabel('Amplitude');
    legend('Suppressed to Dominant Red', 'Dominant to Supressed Blue');
    
    % Plot averaged rivalry traces

    figure
    title([groupPlotLabel ' rivalry: averaged'], 'FontSize', 16)    
    hold on

    mseb([(-transHalf):(1/512):transHalf],[meanTraceLeft; meanTraceRight],[errorTraceLeft; errorTraceRight],[],1);
    
    ylim([-3 3]) %[-1 2])
    %vline(0, 'k', 'button press')
    xlabel('Time from button press (s)');
    ylabel('Amplitude');
    legend('Left Eye', 'Right Eye')
    
    % Plot averaged sim traces
    figure
    title([groupPlotLabel ' simulation: averaged'], 'FontSize', 16)    
    hold on

    mseb([(-transHalf):(1/512):transHalf],[meanTracef1; meanTracef2],[errorTracef1; errorTracef2],[],1);
    
    ylim([-3 3]) %[-1 2])
    %vline(0, 'k', 'button press')
    xlabel('Time from button press (s)');
    ylabel('Amplitude');
    legend('Left Eye', 'Right Eye')
    
    clearvars -except transHalf
end
end