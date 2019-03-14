# input_reader_builder-->dataset_builder.build(input_reader_config, model_config,
                                    training, voxel_generator, target_assigner)


input_reader_builder.build: transform the data(from the .config) into dataset(dict:(-1,9))
dataset_builder.build: transform the data(from the .config) into dict(['voxels', 'num_points', 'coordinates', 'rect', 'Trv2c', 'P2', 'anchors', 'image_idx', 'image_shape'])


if we need to detect it one by one frame, just modify the input_reader_builder.build
