# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.events import EventStorage
import torch
from detectron2.structures import BitMasks
import numpy as np
from PIL import Image

from detectron2.utils.comm import is_main_process



class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass

    def visualization(self, inputs, outputs):
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        with EventStorage() as storage:
            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.time()
                    total_compute_time = 0

                start_compute_time = time.time()
                outputs = model(inputs)
                # outputs = model(inputs, do_postprocess=False)
                # #
                # co = np.array([np.array([0., 0., 1.], dtype=np.float32),
                #                np.array([0., 1., 0.], dtype=np.float32),
                #                np.array([1., 0., 0.], dtype=np.float32),
                #                np.array([0., 1., 1.], dtype=np.float32),
                #                np.array([1., 0., 1.], dtype=np.float32),
                #                np.array([1., 1., 0.], dtype=np.float32),
                #                np.array([0., 0., 0.5], dtype=np.float32),
                #                np.array([0., 0.5, 0.], dtype=np.float32),
                #                np.array([0.5, 0., 0.], dtype=np.float32)])
                # #
                # gt_img = vis_image(inputs[0], inputs[0], mode="gt", colors=co, ind_lst=[0])
                # # vis.images(gt_img, win_name='gt')
                # # im = Image.fromarray(gt_img.transpose(1, 2, 0))
                # # im.save("/root/detectron2/figs/ours_{}_gt.jpg".format(idx))
                #
                # outputs_resize = detector_postprocess(outputs[0], inputs[0]['image'].size(1), inputs[0]['image'].size(2))
                # import ipdb
                # ipdb.set_trace()
                # pred_img = vis_image(inputs[0], outputs_resize, mode="pred", colors=co, ind_lst=[0])

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.time() - start_compute_time

                evaluator.process(inputs, outputs)
                if (idx + 1) % logging_interval == 0:
                    duration = time.time() - start_time
                    seconds_per_img = duration / (idx + 1 - num_warmup)
                    eta = datetime.timedelta(
                        seconds=int(seconds_per_img * (total - num_warmup) - duration)
                    )
                    logger.info(
                        "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        )
                    )
                # if idx > 10:
                #     break
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def vis_image(inputs, instance, mode="gt", colors=None, ind_lst=None):
    v_output = Visualizer(inputs['image'].numpy()[::-1, :, :].transpose(1, 2, 0), None)
    if mode == "gt":
        v_output = v_output.overlay_instances(masks=instance["inference_instances"].gt_masks.tensor[ind_lst], assigned_colors=colors)
    else:
        v_output = v_output.overlay_instances(masks=instance.pred_amodal2_masks[ind_lst].cpu(), assigned_colors=colors)
        # v_output = v_output.overlay_instances(boxes=instance.pred_boxes[ind_lst].tensor.cpu(), assigned_colors=colors)

    img = v_output.get_image()
    img = img.transpose(2, 0, 1)
    vis.images(img, win_name='{}'.format(mode))

    return img


def save_images(gt_img, pred_img, no=0, model_name=None):
    im = Image.fromarray(gt_img.transpose(1, 2, 0))
    im.save("/root/detectron2/figs/supp/gt{}.jpg".format(str(no).zfill(3)))

    im = Image.fromarray(pred_img.transpose(1, 2, 0))
    im.save("/root/detectron2/figs/supp/{}{}.jpg".format(model_name, str(no).zfill(3)))


def embedding_inference_on_train_dataset(model, train_dataloader):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        train_dataloader: an iterable object with a length.
            The elements it generates will be the inputs to the model.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    model.roi_heads.inference_embedding = True
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start embedding inference on training {} images".format(len(train_dataloader)))

    total = len(train_dataloader)  # inference data loader must have a fixed length

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        with EventStorage() as storage:
            for idx, inputs in enumerate(train_dataloader):
                if idx == num_warmup:
                    start_time = time.time()
                    total_compute_time = 0

                start_compute_time = time.time()
                model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.time() - start_compute_time
                if (idx + 1) % logging_interval == 0:
                    duration = time.time() - start_time
                    seconds_per_img = duration / (idx + 1 - num_warmup)
                    eta = datetime.timedelta(
                        seconds=int(seconds_per_img * (total - num_warmup) - duration)
                    )
                    logger.info(
                        "Embedding inference done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        )
                    )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total embedding inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total embedding inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    # model.roi_heads.recon_net.cluster()

    model.roi_heads.inference_embedding = False
    return model


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
