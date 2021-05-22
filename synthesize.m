function output = synthesize(texture_path, nhood_size, output_size, output_path)
% Author: Kai Yang
% @texture_path: path to initial texture
% @nhood_size:   window size for comparison
% @output_size:  size of poutput image
% @output_path:  location to save output image
% synthesize(texture_path, nhood_size, output_size)
%
% Synthesizes an output image if size output_size based on texture given 
% in texture_path. Texture matching is based on nhood_size
%
% Example of using the code:
% >> synthesize('./texture/texture1.jpg', 31, [200, 200], './outputs/texture1_synth.jpg');
%

assert(nhood_size >= 1,'Neighbourhood size is too small');
assert(mod(nhood_size,2)~=0,'Neighbourhood size has to be odd');

% Get texture
text = imread(texture_path);

% Create empty image, output_size = [height, width] or int
assert(length(output_size) < 3, 'Output size not accepted, please give either an Integer or an array of size 2');

% min_size = size of texture_path
init_size = size(text);
assert(output_size(1) >= init_size(1),'Output size should be larger than texture size');
if length(output_size) == 1
    output_size = [output_size, output_size];
end
height = max(output_size(1), init_size(1));
width = max(output_size(2), init_size(2));
channel = 3;
if length(init_size) == 2
    channel = 1;
end
output = ones([height,width,channel]).*-1;

% copy original texture to top left
output(1:init_size(1),1:init_size(2),:) = text;

% Declare utilities
% Create Gauss filter
gauss_filter = fspecial('gaussian',nhood_size,6.4);
if channel == 3
    gauss_filter = repmat(gauss_filter,1,1,3);
end
    % Check for indexes at boundaries
    function [min_i,max_i,min_j,max_j] = get_min_array(i,j,rad,img_size)
       min_i = max(i-rad, 1);
       max_i = min(i+rad, img_size(1));
       min_j = max(j-rad, 1);
       max_j = min(j+rad, img_size(2));
    end
    % Get the number of known neighbours of a pixel within the nhood size
    function n_neighbours = get_n_neighbours(i,j,nhood,img)
        rad = floor(nhood/2);
        [min_i,max_i,min_j,max_j] = get_min_array(i,j,rad,size(img));
        window = img(min_i:max_i,min_j:max_j,1);
        % Sum num of known pixels (not -1)
        n_neighbours = sum(sum(window > -1));
    end
    % Compute similarity between 2 patches
    function sim = get_sim(patch1, patch2, filter)
        % only compare known pixels from patch1
        known_patch1 = (patch1 > -1);
        known_patch2 = (patch2 > -1);
        
        % must have all of patch 1
        common = known_patch1 & known_patch2;
        if sum(sum(sum(common))) < sum(sum(sum(known_patch1)))
            sim = -10000;
            return
        end
        
        patch1 = patch1 .* common;
        patch2 = patch2 .* common;

        % Gaussian SSD
        sq_diff = (patch1 - patch2).^2;
        sq_diff = sq_diff .* filter;
        sim = -sum(sum(sum(sq_diff)));
        
        if sim == 0
            sim = -10000;
        end
    end
    % Find unknown neighbours given a coordinate
    function neighbours = get_neighbours(i,j,nhood,img)
        [min_i,max_i,min_j,max_j] = get_min_array(i,j,1,size(img));
        num_unknown_neighbours = sum(sum(img(min_i:max_i,min_j:max_j,1)==-1));
        % return: [i,j,num_known_neighbours]
        neighbours = ones([num_unknown_neighbours,3]);
        count = 1;
        for row = min_i:max_i
            for col = min_j:max_j
                % if not yet filled
                if img(row,col,1) == -1
                    neighbours(count,:) = [row, col, get_n_neighbours(row,col,nhood,img)];
                    count = count + 1;
                end
            end
        end
    end
    % return window, account for boundary pixel
    function window = get_window(i,j,nhood,img)
        rad = floor(nhood/2);
        img_size = size(img);
        [min_i,max_i,min_j,max_j] = get_min_array(i,j,rad,img_size);
        % pixel is near boundary
        n_channel = 3;
        if length(img_size) == 2
            n_channel = 1;
        else
            n_channel = img_size(3);
        end
        window = ones([nhood,nhood,n_channel]).*-1;
        if ((max_i - min_i) < nhood - 1) || ((max_j - min_j) < nhood - 1)
            % pixel is at boundary, the rest will be -1 (not considered)
            idx_i = i - min_i;
            idx_j = j - min_j;
            diff_i = max_i-min_i;
            diff_j = max_j-min_j;
            start_i = rad - idx_i + 1;
            start_j = rad - idx_j + 1;
            end_i = start_i + diff_i;
            end_j = start_j + diff_j;
            window(start_i:end_i,start_j:end_j,:) = img(min_i:max_i,min_j:max_j,:);
        else
            window(:,:,:) = img(min_i:max_i,min_j:max_j,:);
        end
    end
    % Vectorized Approach
    % Search and return the pixel with the most similar neighbourhood
    % Stack all windows from init texture
    num_pixels = init_size(1)*init_size(2)
    stacked_text = zeros([num_pixels,nhood_size,nhood_size,channel]);
    for text_row = 1:init_size(1)
        for text_col = 1:init_size(2)
            pixel_val = (text_row-1)*init_size(1) + text_col;
            wind = get_window(text_row, text_col, nhood_size, text);
            stacked_text(pixel_val,:,:,:) = wind;
        end
    end
    function [best_i, best_j] = search_stack(i,j,nhood,img,stack,filter,init_size)
        % early exit if alr assigned
        if img(i,j,1) ~= -1
            best_i = -1;
            best_j = -1;
            return;
        end
        % Get window around pixel
        unknown_window = get_window(i,j,nhood,img);
        window_size = size(unknown_window);
        w_channel = 3;
        if length(window_size) == 2
            w_channel = 1;
        end
        unknown_window = reshape(unknown_window,[],window_size(1),window_size(2),w_channel); 
        
        % only compare known pixels from unknown_patch
        known = (unknown_window > -1);  % Binary array
        known_stack = (stack > -1);     % Binary array
        common_stack = (known .* known_stack); % Binary array
       
        % get sq diff across all windows 
        stack = common_stack .* stack;      % keep known values
        known_window = known .* unknown_window;  % keep known values
        sq_diff = (stack - known_window).^2;
        filter = reshape(filter,[],window_size(1),window_size(2),w_channel); 
        sq_diff = sq_diff .* filter;
        sim = reshape(sum(sum(sum(sq_diff,2),3),4),[],1);
        
        % remove irrelevant windows by adding arbitrary loss
        irrelevant = sum(sum(sum(common_stack,2),3),4) < sum(sum(sum(known)));
        irrelevant = irrelevant  .* 100000;
        sim = sim + irrelevant;

        % find most similar
        [min_val, best_idx] = min(sim);
        best_i = floor((best_idx-1)/(init_size(1))) + 1;
        best_j = mod(best_idx, init_size(1));
        if best_j == 0
            best_j = init_size(2);
        end
        
    end
    % Find best texture match by Iterate through all windows
    function [best_i, best_j] = get_best_match(i,j,nhood,img,texture,filter)
        % early exit if alr assigned
        if img(i,j,1) ~= -1
            best_i = -1;
            best_j = -1;
            return;
        end
        text_size = size(texture);
        best_sim = -10000;
        % Get window around pixel
        unknown_window = get_window(i,j,nhood,img);
        % Search for most similar window in texture
        for row = 1:text_size(1)
            for col = 1:text_size(2)
                text_window = get_window(row,col,nhood,texture);
                sim = get_sim(unknown_window, text_window, filter);
                if sim > best_sim
                    best_sim = sim;
                    best_i = row;
                    best_j = col;
                end
            end
        end
    end
% [ Implemented a min heap below instead ]
% Using naive method of picking next pixel with most neighbours
%     function q = insert_pixel(num_neighbours, i, j, q)
%         key = string(i) + ' ' + string(j);
%         % if is key, check if need to update
%         if isKey(q, key)
%             if q(key) < num_neighbours
%                 q(key) = num_neighbours;
%             end
%         % Otherwise, just add    
%         else
%             q(key) = num_neighbours;
%         end
%     end
%     function [q, next_pixel] = get_next_pixel(q)
%         most_neighbours = 0;
%         q_keys = keys(q);
%         key = '';
%         % Iterate through list of keys
%         for k = 1:length(q)
%             if q(q_keys{k}) > most_neighbours
%                 most_neighbours = q(q_keys{k});
%                 key = q_keys{k};
%             end
%         end
%         % remove entry from q
%         remove(q, key);
%         next_pixel = split(key,' ');
%         next_pixel = [str2double(next_pixel{1}) str2double(next_pixel{2})];
%     end

% TEST function
    % Visually compare original window and matched window
    function test(next_pixel, best_match, nhood, img, text)
        init_window = get_window(next_pixel(1),next_pixel(2),nhood,img);
        matched_window = get_window(best_match(1),best_match(2),nhood,text);
        figure;
        imshow(cast(init_window,'uint8'));
        title('init');
        figure;
        imshow(cast(matched_window,'uint8'));
        title('matched');
    end

% Min Priority Queue implementation
    function new_heap = extend_list(heap)
        heap_size = size(heap);
        new_heap = zeros([heap_size(1)*2 heap_size(2)]);
        new_heap(1:length(heap),:) = heap;
    end
    function heap = swap(heap, cur_idx, new_idx)
        temp = heap(cur_idx,:);
        heap(cur_idx,:) = heap(new_idx,:);
        heap(new_idx,:) = temp;
    end
    function [last_idx,heap] = insert(val,i,j,heap,last_idx)
        last_idx = last_idx + 1;
        if last_idx > length(heap)
            heap = extend_list(heap);
        end
        heap(last_idx,:) = [val i j];
        parent = floor(last_idx/2);
        if parent < 1
            return
        end
        cur_idx = last_idx;
        while (heap(parent,1) > heap(cur_idx,1)) && (cur_idx ~= 1)
            heap = swap(heap, cur_idx, parent);
            cur_idx = parent;
            parent = floor(cur_idx/2);
            if parent < 1
                return
            end
        end    
    end
    function [val,last_idx, heap] = extract(heap, last_idx)
        val = heap(1,:);
        % replace root with last element
        heap = swap(heap, last_idx, 1);
        last_idx = last_idx - 1;
        % swap down
        heap = shift_down(heap, 1, last_idx);
    end
    function heap = shift_down(heap, idx, last_idx)
        stop = 0;
        while (idx < last_idx) && (stop == 0)
            min_V = heap(idx,1);
            min_idx = idx;
            left = idx * 2;
            right = idx * 2 + 1;
            % If left child smaller
            if (left <= last_idx) && (min_V > heap(left,1))
                min_V = heap(left,1);
                min_idx = left;
            end
            % if right child smaller
            if (right <= last_idx) && (min_V > heap(right,1))
                min_V = heap(right,1);
                min_idx = right;
            end
            if (min_idx ~= idx)
                heap = swap(heap, idx, min_idx);
                idx = min_idx;
            else
                stop = 1;
            end
        end        
    end

% Texture Synthesis

pixel_heap = zeros([8,3]); %% init heap with size 8
last_idx = 0;
% Insert first few candidates (those right beside the initial texture
for o_row = 2:output_size(1)
    for o_col = 2:output_size(2)
        % filter [X -1] and [[X] [-1]]
        if ((output(o_row,o_col-1,1) > -1) && (output(o_row,o_col,1) == -1)) ||...
                ((output(o_row-1,o_col,1) > -1) && (output(o_row,o_col,1) == -1))
            % Add to q
            n_neighbours = get_n_neighbours(o_row,o_col,nhood_size,output);
            [last_idx, pixel_heap] = insert(-n_neighbours,o_row,o_col,pixel_heap,last_idx);
        end
    end
end

total_to_fill = output_size(1) * output_size(2) - init_size(1) * init_size(2);
filled = 0;

tic 
% Start texture_synthesis
while last_idx > 0
    % Get next
    [val,last_idx, pixel_heap] = extract(pixel_heap,last_idx);
    next_pixel = val(2:3);
    % Find best match
    % Using normal search across texture
    %[best_i, best_j] = get_best_match(next_pixel(1), next_pixel(2), nhood_size, output, text, gauss_filter);
    % Using stacked windows for search (Vectorized)
    [best_i, best_j] = search_stack(next_pixel(1),next_pixel(2),nhood_size,output,stacked_text,gauss_filter,init_size);
    % Only consider if not already filled
    if best_i ~= -1
        filled = filled + 1;
        % Replace
        output(next_pixel(1),next_pixel(2),:) = text(best_i,best_j,:);
        % Add new neighbours
        neighbours = get_neighbours(next_pixel(1),next_pixel(2),nhood_size,output);
        for n = 1:length(neighbours(:,1))
            n_neighbours = neighbours(n,3);
            [last_idx, pixel_heap] = insert(-n_neighbours,neighbours(n,1),neighbours(n,2),pixel_heap,last_idx);
        end
        
        % To display initial window and matched window from texture
        % test(next_pixel, [best_i, best_j], nhood_size, output, text);
        
        % Uncomment to display progress
        % imshow(cast(output,'uint8'))
        
        % log progress
        if mod(filled,100) == 0
            string(string(filled)+' / '+string(total_to_fill)) 
            %imshow(cast(output,'uint8'))
        end
    end
end
toc
 
output = cast(output,'uint8');
imshow(output)
imwrite(output, output_path);
end