import logging

def configure_logger(verbose=False):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    return logger

def log_time(start, batch_end_time, n_batches=None, ind=None):
        if ind is not None or n_batches is not None:
            elapsed_time = batch_end_time - start
            avg_time_per_batch = elapsed_time / (ind + 1)
            remaining_batches = n_batches - (ind + 1)
            estimated_remaining_time = remaining_batches * avg_time_per_batch
            hours, remainder = divmod(estimated_remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
        else:
            endtime = batch_end_time
            dur = endtime - start
            hours, remainder = divmod(dur, 3600)
            minutes, seconds = divmod(remainder, 60)
        return hours, minutes, seconds


