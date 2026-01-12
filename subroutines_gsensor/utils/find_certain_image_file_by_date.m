function next_image_file = find_certain_image_file_by_date(folder, ptr)
    next_image_file = [];
    image_files = find_image_files_by_date(folder);
    try
        next_image_file = image_files(ptr);
    catch
    end
end