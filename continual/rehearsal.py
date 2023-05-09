import copy

import numpy as np
import torch
from torchvision import transforms

from continual.mycontinual import ArrayTaskSet


class Memory:
    def __init__(self, memory_size, nb_total_classes, rehearsal, rep_replay=False, fixed=True):
        self.memory_size = memory_size # 2000
        self.nb_total_classes = nb_total_classes # 100
        self.rehearsal = rehearsal # icarl_all
        self.fixed = fixed # False
        self.rep_replay = rep_replay

        self.x = self.y = self.t = None

        self.nb_classes = 0

    @property
    def memory_per_class(self):
        if self.fixed:
            return self.memory_size // self.nb_total_classes
        return self.memory_size // self.nb_classes if self.nb_classes > 0 else self.memory_size

    def get_dataset_without_copy(self, base_dataset):
        dataset = base_dataset
        dataset._x = self.x
        dataset._y = self.y
        dataset._t = self.t

        return dataset

    def get_dataset(self, base_dataset):
        if self.rep_replay:
            dataset = ArrayTaskSet(x=self.x, y=self.y, t=self.t, trsf=None, target_trsf=None,
                                   bounding_boxes=None)
        else:
            dataset = copy.deepcopy(base_dataset)
            dataset._x = self.x
            dataset._y = self.y
            dataset._t = self.t

        return dataset

    def get(self):
        return self.x, self.y, self.t

    def __len__(self):
        return len(self.x) if self.x is not None else 0

    def save(self, path):
        np.savez(
            path,
            x=self.x, y=self.y, t=self.t
        )

    def load(self, path):
        data = np.load(path)
        self.x = data["x"]
        self.y = data["y"]
        self.t = data["t"]

        assert len(self) <= self.memory_size, len(self)
        self.nb_classes = len(np.unique(self.y))

    def reduce(self):
        x, y, t = [], [], []
        for class_id in np.unique(self.y):
            indexes = np.where(self.y == class_id)[0]
            x.append(self.x[indexes[:self.memory_per_class]])
            y.append(self.y[indexes[:self.memory_per_class]])
            t.append(self.t[indexes[:self.memory_per_class]])

        self.x = np.concatenate(x)
        self.y = np.concatenate(y)
        self.t = np.concatenate(t)

    def add(self, dataset, model, nb_new_classes):
        self.nb_classes += nb_new_classes

        x, y, t = herd_samples(dataset, model, self.memory_per_class, self.rehearsal, self.rep_replay)

        if self.x is None:
            self.x, self.y, self.t = x, y, t
        else:
            if not self.fixed:
                self.reduce()
            self.x = np.concatenate((self.x, x))
            self.y = np.concatenate((self.y, y))
            self.t = np.concatenate((self.t, t))


def herd_samples(dataset, model, memory_per_class, rehearsal, rep_replay):
    x, y, t = dataset._x, dataset._y, dataset._t

    if rehearsal == "random":
        indexes = []
        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            indexes.append(
                np.random.choice(class_indexes, size=memory_per_class)
            )
        indexes = np.concatenate(indexes)

        return x[indexes], y[indexes], t[indexes]
    elif "closest" in rehearsal:
        if rehearsal == 'closest_token':
            handling = 'last'
        else:
            handling = 'all'

        features, targets = extract_features(dataset, model, handling)
        indexes = []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            class_features = features[class_indexes]

            class_mean = np.mean(class_features, axis=0, keepdims=True)
            distances = np.power(class_features - class_mean, 2).sum(-1)
            class_closest_indexes = np.argsort(distances)

            indexes.append(
                class_indexes[class_closest_indexes[:memory_per_class]]
            )

        indexes = np.concatenate(indexes)
        return x[indexes], y[indexes], t[indexes]
    elif "furthest" in rehearsal:
        if rehearsal == 'furthest_token':
            handling = 'last'
        else:
            handling = 'all'

        features, targets = extract_features(dataset, model, handling)
        indexes = []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            class_features = features[class_indexes]

            class_mean = np.mean(class_features, axis=0, keepdims=True)
            distances = np.power(class_features - class_mean, 2).sum(-1)
            class_furthest_indexes = np.argsort(distances)[::-1]

            indexes.append(
                class_indexes[class_furthest_indexes[:memory_per_class]]
            )

        indexes = np.concatenate(indexes)
        return x[indexes], y[indexes], t[indexes]
    elif "icarl":
        if rehearsal == 'icarl_token':
            handling = 'last'
        else:
            handling = 'all'

        features, targets = extract_features(dataset, model, handling)
        indexes = []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            class_features = features[class_indexes]

            indexes.append(
                class_indexes[icarl_selection(class_features, memory_per_class)]
            )

        indexes = np.concatenate(indexes)

        # store representations for the samples
        if rep_replay:

            dataset.trsf = transforms.Compose([dataset.trsf.transforms[tf] for tf in [0,3,4]])
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                num_workers=2,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )

            features, targets = [], []

            with torch.no_grad():
                for x, y, _ in loader:
                    if hasattr(model, 'module'):
                        reps = model.module.forward_initial(x.cuda())
                    else:
                        reps = model.forward_initial(x.cuda())

                    reps = reps.detach().cpu().numpy()
                    y = y.numpy()

                    features.append(reps.reshape((reps.shape[0], int(reps.shape[1] ** 0.5),
                                                               int(reps.shape[1] ** 0.5), reps.shape[-1])))
                    targets.append(y)

            features = np.vstack(features)
            targets = np.concatenate(targets)

            return features[indexes], targets[indexes], t[indexes]
        else:
            return x[indexes], y[indexes], t[indexes]
    else:
        raise ValueError(f"Unknown rehearsal method {rehearsal}!")


def extract_features(dataset, model, ensemble_handling='last'):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    features, targets = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            if hasattr(model, 'module'):
                feats, _, _ = model.module.forward_features(x.cuda())
            else:
                feats, _, _ = model.forward_features(x.cuda())

            if isinstance(feats, list):
                if ensemble_handling == 'last':
                    feats = feats[-1]
                elif ensemble_handling == 'all':
                    feats = torch.cat(feats, dim=1)
                else:
                    raise NotImplementedError(f'Unknown handling of multiple features {ensemble_handling}')
            elif len(feats.shape) == 3:  # joint tokens
                if ensemble_handling == 'last':
                    feats = feats[-1]
                elif ensemble_handling == 'all':
                    feats = feats.permute(1, 0, 2).view(len(x), -1)
                else:
                    raise NotImplementedError(f'Unknown handling of multiple features {ensemble_handling}')

            feats = feats.cpu().numpy()
            y = y.numpy()

            features.append(feats)
            targets.append(y)

    features = np.concatenate(features)
    targets = np.concatenate(targets)

    return features, targets


def icarl_selection(features, nb_examplars):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]


def get_finetuning_dataset(dataset, memory, finetuning='balanced', rep_replay=False):
    if finetuning == 'balanced':
        x, y, t = memory.get()

        if rep_replay:
            # current task samples
            new_dataset = ArrayTaskSet(x=x, y=y, t=t, trsf=None,
                                         target_trsf=None, bounding_boxes=None)
        else:
            new_dataset = copy.deepcopy(dataset)
            new_dataset._x = x
            new_dataset._y = y
            new_dataset._t = t
    elif finetuning in ('all', 'none'):
        new_dataset = dataset
    else:
        raise NotImplementedError(f'Unknown finetuning method {finetuning}')

    return new_dataset


def get_separate_finetuning_dataset(dataset, memory, finetuning='balanced', rep_replay=False):
    if finetuning == 'balanced':
        x, y, t = memory.get()

        # extract current and old task samples from memory
        cur_task_idx = t == max(np.unique(t))
        old_task_idx = t != max(np.unique(t))

        if rep_replay:
            # current task samples
            first_dataset = ArrayTaskSet(x=x[cur_task_idx], y=y[cur_task_idx], t=t[cur_task_idx], trsf=None,
                                         target_trsf=None, bounding_boxes=None)

            # old task samples
            second_dataset = ArrayTaskSet(x=x[old_task_idx], y=y[old_task_idx], t=t[old_task_idx], trsf=None,
                                          target_trsf=None, bounding_boxes=None)
        else:
            first_dataset = copy.deepcopy(dataset)
            first_dataset._x = x[cur_task_idx]
            first_dataset._y = y[cur_task_idx]
            first_dataset._t = t[cur_task_idx]

            second_dataset = copy.deepcopy(dataset)
            second_dataset._x = x[old_task_idx]
            second_dataset._y = y[old_task_idx]
            second_dataset._t = t[old_task_idx]

    elif finetuning in ('all', 'none'):
        # not supported after change
        new_dataset = dataset
    else:
        raise NotImplementedError(f'Unknown finetuning method {finetuning}')

    return first_dataset, second_dataset
