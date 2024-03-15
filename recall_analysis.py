import os


def fashioniq(root):
    type = ['dress', 'shirt', 'toptee']
    dict = {}
    for t in type:
        recall_path = os.path.join(root, 'recall_result_fiq_'+t+'.txt')
        dict[t] = []
        with open(recall_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split("\t")
                r_at10 = round(float(l[2].replace('R@10: ', '')), 3)
                r_at50 = round(float(l[3].replace('R@50: ', '')), 3)
                dict[t].append([r_at10, r_at50])

    alltype_recall = zip(dict['dress'], dict['shirt'], dict['toptee'])
    avgRecall = []
    for one_epoch in alltype_recall:
        R_at10 = [x[0] for x in one_epoch]
        R_at50 = [x[1] for x in one_epoch]
        avgR_at10 = round(sum(R_at10) / 3, 3)
        avgR_at50 = round(sum(R_at50) / 3, 3)
        avgR = round((avgR_at10 + avgR_at50) / 2, 3)
        avgRecall.append(avgR)

    return avgRecall

def cirr(root):
    recall_path = os.path.join(root, 'recall_result_cirr.txt')
    avg_recall = []
    with open(recall_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.split("\t")
            avg_recall.append(round(float(l[8].replace('Avg_R: ', '')), 3))
                   
    return avg_recall

def coco(root):
    recall_path = os.path.join(root, 'recall_result_coco.txt')
    avg_recall = []
    with open(recall_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.split("\t")
            avg_recall.append(round(float(l[5].replace('ft_Avg_R: ', ''))*100, 3))
                   
    return avg_recall

def imgnet(root):
    recall_path = os.path.join(root, 'recall_result_imgnet.txt')
    avg_recall = []
    with open(recall_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip("\n").split("\t")
            r_at10 = l[1::3]
            r_at50 = l[2::3]
            r_at10 = [float(x.replace('R@10: ', '')) for x in r_at10]
            r_at50 = [float(y.replace('R@50: ', '')) for y in r_at50]
            avgR_at10 = sum(r_at10) / len(r_at10)
            avgR_at50 = sum(r_at50) / len(r_at50)
            avgR = round((avgR_at10 + avgR_at50) / 2 * 100, 3)
            avg_recall.append(avgR)
    return avg_recall

if __name__ == "__main__":
    root_path = '/home/jumpserver/yxt/cir/composed_image_retrieval/logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-03-13-00-20-38'
    
    faiq_avg_recall = fashioniq(root_path)
    cirr_avg_recall = cirr(root_path)
    coco_avg_recall = coco(root_path)
    imgn_avg_recall = imgnet(root_path)
    
    composed_recall = zip(faiq_avg_recall, cirr_avg_recall, coco_avg_recall, imgn_avg_recall)
    composed_avg_recall = []
    i = 1
    for c in composed_recall:
        print(f"{i}\t{c}")
        i += 1
        composed_avg_recall.append(round(sum(c) / len(c), 3))

    max_index = composed_avg_recall.index(max(composed_avg_recall))

    print(f"faiq\tepoch {1 + faiq_avg_recall.index(max(faiq_avg_recall))}\t{max(faiq_avg_recall)}")
    print(f"cirr\tepoch {1 + cirr_avg_recall.index(max(cirr_avg_recall))}\t{max(cirr_avg_recall)}")
    print(f"coco\tepoch {1 + coco_avg_recall.index(max(coco_avg_recall))}\t{max(coco_avg_recall)}")
    print(f"imgn\tepoch {1 + imgn_avg_recall.index(max(imgn_avg_recall))}\t{max(imgn_avg_recall)}")
    print(f"COMP\tepoch {1 + max_index}\t{max(composed_avg_recall)}\t\
          {faiq_avg_recall[max_index]}\t{cirr_avg_recall[max_index]}\t{coco_avg_recall[max_index]}\t{imgn_avg_recall[max_index]}")