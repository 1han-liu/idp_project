function last_image_file = find_last_image_file_by_date(folder)
    image_files = find_image_files_by_date(folder);
    if ~isempty(image_files)
        last_image_file = image_files(end);
    else
        last_image_file = [];
    end
end