function fist_image_file = find_first_image_file_by_date(folder)
    image_files = find_image_files_by_date(folder);
    if ~isempty(image_files)
        fist_image_file = image_files(1);
    else
        fist_image_file = [];
    end
end