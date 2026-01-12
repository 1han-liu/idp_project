function files = find_image_files_by_date(folder)
    files = dir(fullfile(folder, 'IMG_*.png'));
    [~, idx] = sort([files.datenum]);
    files = files(idx);
end