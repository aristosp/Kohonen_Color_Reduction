% © Aristeidis Papadopoulos AEM: 57576
clc
clear
close all
tic
% Display Example Images
figure
subplot(2,3,1)
imshow(imread('toy.jpg')); title('Image 1');
subplot(2,3,2)
imshow(imread('flowers.jfif')); title('Image 2');
subplot(2,3,3)
imshow(imread('UKFlag.jpg')); title('Image 3');
subplot(2,3,4)
imshow(imread('StevieWonder_Superstition.jpg')); title('Image 4');
subplot(2,3,5)
imshow(imread('FooFighters_WastingLight.jpg')); title('Image 5');
subplot(2,3,6)
imshow(imread('KnossosPainting.jpg')); title('Image 6');
disp('If user selection is invalid, image 1 will be used')
image_select = input('Select Image To Quantize: ');% Save user's selection

if image_select == 1
        image = imread('toy.jpg');
elseif image_select == 2
        image = imread('flowers.jfif');
elseif image_select == 3
        image = imread('UKFlag.jpg');
elseif image_select == 4
        image = imread('StevieWonder_Superstition.jpg');
elseif image_select == 5
        image = imread('FooFighters_WastingLight.jpg');
elseif image_select == 6
        image = imread('KnossosPainting.jpg');
else
        disp('Error! User has not selected an image.Default image will be used');
        image = imread('toy.jpg'); % Default choice
end

close all %Close all windows

imshow(image);title('Original Image');% Display Original Image
disp('If user input is invalid, then default value of 10 colors will be used.');
neurons = input('Enter number of colors in quantized image: '); % User selection of output colors
if neurons == 0 % Check is user selected invalid number of output colors
    neurons = 10; % Determine the number of output neurons and so the final colors
end

num_of_samples = 5000;% Number of training samples
epochs = 10; % Number of training iterations
lr = 0.1; % Learning Rate
neighborhood_radius = (2/3) * neurons; % Determine starting neighborhood radius for winner neuron

x_train = randi(size(image,1),[num_of_samples,1]); % Randomized Pixel coordinates in x axis
y_train = randi(size(image,2),[num_of_samples,1]); % Randomized Pixel coordinates in y axis
samples = zeros(num_of_samples,3);

for i = 1:num_of_samples
    samples(i,:) = double(image(x_train(i),y_train(i),:)); 
    % Save rgb values of each sample pixel
end

weights = zeros(neurons,3);% Neuron matrix initialization

for i = 1:neurons
    % Randomly Initialize Neuron weights from the available samples
    weights(i,:) = samples(randi([0,num_of_samples]),:);
end

t = 0;% Starting epoch

% Training
disp('Training has started...')
while t <= epochs
    lr = lr * (1 - t/epochs); % Update learning rate
    neighborhood_radius = round(neighborhood_radius * (1 - t/epochs)); % Update neighborhood radius
    for j = 1:num_of_samples
            % Calculate euclidean distance between sample RGB and neurons
            eu_dist = pdist2(weights,samples(j,:));
            [~,idx] = min(eu_dist); % Find winner neuron
            neighbor_dist = pdist2(weights(idx,:),weights); % Find nearest neighbors
            [neighbors_sorted,indexes] = sort(neighbor_dist); % Sort them by ascending order
            % Competitive Learning Rule
            weights(idx,:) = weights(idx,:) + lr * (samples(j,:) - weights(idx,:)); % Update winner neuron
            for z = 1:neighborhood_radius % Update the rest of the neighborhood
                if neighbors_sorted(z) ~= 0 % Exclude winning neuron
                    b = exp(- neighbors_sorted(z)); % Neighborhood Coefficient
                    weights(indexes(z),:) = weights(indexes(z),:) + lr * b * (samples(j,:) - weights(indexes(z),:));
                end
            end      
    end
    t = t + 1; % Increment epoch
end
disp('Training has ended.')
new_img = image;

% Image Traversal
disp('Replacing the pixels of the original image...')
for y = 1:size(image,1)
    for x = 1:size(image,2)
         current_RGB = double(image(y,x,:));% Get current RGB Values
         % Create the RGB Vector
         current_RGB = reshape(current_RGB,1,3);
         dist = pdist2(weights,current_RGB);% Calculate euclidean distance between RGB Vector and neurons
         [~,index] = min(dist);% Find minimum distance neuron
         % Replace the RGB Values of the original image to the neuron ones
         new_img(y, x, : ) = uint8(weights(index,:));
    end
end

disp('New image has been made,creating appropriate plots...')
colors_in_new_image = uint8(reshape(weights,size(weights,1),1,3)); % Create the color palette of the new image
figure
subplot(1,2,1)
imshow(new_img); title('New Image with Reduced Colors');
subplot(1,2,2)
imshow(colors_in_new_image, 'InitialMagnification', 'fit'); title('Color Palette in New Image');
disp('Program has ended.')
toc