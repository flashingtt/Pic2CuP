import os
import PIL
from PIL import Image
from tqdm import tqdm

def fashioniq(best_dir, compared_dir):
    dataset_dir = '/home/jumpserver/yxt/cir/datasets/fashionIQ/images'
    style = ['dress', 'shirt', 'toptee']
    
    for s in style:
        best_res = []
        compared_res = []

        best_res_path = os.path.join(best_dir, 'fashioniq_' + s + '.txt')
        compared_res_path = os.path.join(compared_dir, 'fashioniq_' + s + '.txt')

        style_dir = os.path.join('./visualization_output/fashioniq', s)
        if not os.path.exists(style_dir):
            os.makedirs(style_dir)

        with open(best_res_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                best_res.append(line)
        with open(compared_res_path, 'r') as g:
            lines = g.readlines()
            for line in lines:
                compared_res.append(line)
        assert len(best_res) == len(compared_res)
        print(f"\n{s}\nBest R@1 num: {len(best_res)}")
        count = 0
        repeat = 0
        for i in tqdm(range(len(best_res))):
            best_i = best_res[i].split("\t")
            compared_i = compared_res[i].split("\t")
            if best_i[1] != compared_i[1]:
                count += 1
                assert best_i[0] == compared_i[0]
                ref_name = best_i[0].replace(':', '')
                best_tar_name = best_i[1:5]
                best_tar_name.append(best_i[5].replace("\n", ""))

                compared_tar_name = compared_i[1:5]
                compared_tar_name.append(compared_i[5].replace("\n", ""))

                ref_image = Image.open(os.path.join(dataset_dir, ref_name + '.png'))

                save_dir = os.path.join(style_dir, ref_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                else:
                    save_dir = os.path.join(style_dir, ref_name + '_' + str(repeat))
                    os.makedirs(save_dir)
                    repeat += 1
                    
                best_save_dir = os.path.join(save_dir, 'best')
                os.makedirs(best_save_dir)
                compared_save_dir = os.path.join(save_dir, 'compared')
                os.makedirs(compared_save_dir)

                ref_image.save(os.path.join(save_dir, ref_name + '.png'))

                for i in range(len(best_tar_name)):
                    tar_image = Image.open(os.path.join(dataset_dir, best_tar_name[i] + '.png'))
                    tar_image.save(os.path.join(best_save_dir, best_tar_name[i] + '_top_' + str(i + 1) + '.png'))

                for i in range(len(compared_tar_name)):
                    tar_image = Image.open(os.path.join(dataset_dir, compared_tar_name[i] + '.png'))
                    tar_image.save(os.path.join(compared_save_dir, compared_tar_name[i] + '_top_' + str(i + 1) + '.png'))
                
        print(f"Our model performs better on {count} datas.")


def cirr(best_dir, compared_dir):
    dataset_dir = '/home/jumpserver/yxt/cir/datasets/cirr/dev'
    
    best_res = []
    compared_res = []
    
    best_res_path = os.path.join(best_dir, 'cirr.txt')
    compared_res_path = os.path.join(compared_dir, 'cirr.txt')

    with open(best_res_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            best_res.append(line)
    with open(compared_res_path, 'r') as g:
        lines = g.readlines()
        for line in lines:
            compared_res.append(line)
    assert len(best_res) == len(compared_res)
    print(f"\nBest R@1 num: {len(best_res)}")
    count = 0
    repeat = 0
    
    for i in tqdm(range(len(best_res))):
        best_i = best_res[i].split("\t")
        compared_i = compared_res[i].split("\t")
        if best_i[1] != compared_i[1]:
            count += 1
            assert best_i[0] == compared_i[0]
            ref_name = best_i[0].replace(':', '')
            best_tar_name = best_i[1:5]
            best_tar_name.append(best_i[5].replace("\n", ""))

            compared_tar_name = compared_i[1:5]
            compared_tar_name.append(compared_i[5].replace("\n", ""))

            ref_image = Image.open(os.path.join(dataset_dir, ref_name + '.png'))

            save_dir = os.path.join('./visualization_output/cirr', ref_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                save_dir = os.path.join('./visualization_output/cirr', ref_name + '_' + str(repeat))
                os.makedirs(save_dir)
                repeat += 1
                
            best_save_dir = os.path.join(save_dir, 'best')
            os.makedirs(best_save_dir)
            compared_save_dir = os.path.join(save_dir, 'compared')
            os.makedirs(compared_save_dir)

            ref_image.save(os.path.join(save_dir, ref_name + '.png'))

            for i in range(len(best_tar_name)):
                tar_image = Image.open(os.path.join(dataset_dir, best_tar_name[i] + '.png'))
                tar_image.save(os.path.join(best_save_dir, best_tar_name[i] + '_top_' + str(i + 1) + '.png'))

            for i in range(len(compared_tar_name)):
                tar_image = Image.open(os.path.join(dataset_dir, compared_tar_name[i] + '.png'))
                tar_image.save(os.path.join(compared_save_dir, compared_tar_name[i] + '_top_' + str(i + 1) + '.png'))
            
    print(f"Our model performs better on {count} datas.")


def main():
    best_dir = '/home/jumpserver/yxt/cir/composed_image_retrieval/logs/2024-03-04-16-07-24_best/visual_result_best'
    compared_dir = '/home/jumpserver/yxt/cir/composed_image_retrieval/visual_compared_result'
    # fashioniq(best_dir, compared_dir)
    cirr(best_dir, compared_dir)
    # return 0

if __name__ == '__main__':
    main()
