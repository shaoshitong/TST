import time
import logging
logger=logging.getLogger('DenseNet')
class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"


class Log:
    def __init__(self, log_each: int, initial_epoch=-1):
        self.loading_bar = LoadingBar(length=27)
        self.top_1 = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()
        self.is_train = True
        self.last_steps_state = {"loss": 0.0, "top_1": 0.0,"top_5":0.0 ,"steps": 0}
        self._reset(len_dataset)
    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, model, loss,top_1,top_5, learning_rate: float = None) -> None:
        if self.is_train:
            self._train_step(model, loss,top_1,top_5, learning_rate)
        else:
            self._eval_step(loss,top_1,top_5)

    def flush(self) -> None:
        if hasattr(self,"is_train"):
            if self.is_train:
                loss = self.epoch_state["loss"] / self.epoch_state["steps"]
                top_1 = self.epoch_state["top_1"] / self.epoch_state["steps"]
                top_5 = self.epoch_state["top_5"] / self.epoch_state["steps"]
                print(
                    f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{top_1:11.3f} % │{top_5:11.3f} % ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                    end="",
                    flush=True,
                )
                logger.info(f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{top_1:11.3f} % │{top_5:11.3f} % ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",)

            else:
                loss = self.epoch_state["loss"] / self.epoch_state["steps"]
                top_1 = self.epoch_state["top_1"] / self.epoch_state["steps"]
                top_5 = self.epoch_state["top_5"] / self.epoch_state["steps"]
                print(f"{loss:12.4f}  │{top_1:10.3f} %  │{top_5:10.3f} %  ┃", flush=True)
                logger.info(f"{loss:12.4f}  │{top_1:10.3f} %  │{top_5:10.3f} %  ┃")

                if top_1 > self.top_1:
                    self.top_1 = top_1

    def _train_step(self, model, loss, top_1,top_5, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.last_steps_state["loss"] += loss.item()
        self.last_steps_state["top_1"] += top_1.item()
        self.last_steps_state["top_5"] += top_5.item()
        self.last_steps_state["steps"] += 1
        self.epoch_state["loss"] += loss.item()
        self.epoch_state["top_1"] += top_1.item()
        self.epoch_state["top_5"] += top_5.item()
        self.epoch_state["steps"] += 1
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.last_steps_state["loss"] / self.last_steps_state["steps"]
            top_1 = self.last_steps_state["top_1"] / self.last_steps_state["steps"]
            top_5 = self.last_steps_state["top_5"] / self.last_steps_state["steps"]
            self.last_steps_state = {"loss": 0.0, "top_1": 0.0,"top_5":0.0,"steps": 0}
            progress = self.step / self.len_dataset
            print(self.epoch,loss,top_1,top_5,learning_rate,self._time(),self.loading_bar(progress))
            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{top_1:11.3f} % │{top_5:11.3f} % ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}",
                end="",
                flush=True,
            )
            # logger.info(f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{top_1:11.3f} % │{top_5:11.3f} % ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}")

    def _eval_step(self, loss, top_1,top_5) -> None:
        self.epoch_state["loss"] += loss.item()
        self.epoch_state["top_1"] += top_1.item()
        self.epoch_state["top_5"] += top_5.item()
        self.epoch_state["steps"] += 1

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "top_1": 0.0, "top_5":0.0,"steps": 0}
    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━━━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━━━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━━━━━━━━┓")
        print(f"┃              ┃              ╷              ╷              ┃              ╷              ┃              ╷              ╷              ┃")
        print(f"┃       epoch  ┃        loss  │    top_1     ╷    top_5     ┃        l.r.  │     elapsed  ┃        loss  │     top_1    │     top_5    ┃")
        print(f"┠──────────────╂──────────────┼──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┼──────────────┨")
        logger.info(f"┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━━━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━━━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━━━━━━━━┓")
        logger.info(f"┃              ┃              ╷              ╷              ┃              ╷              ┃              ╷              ╷              ┃")
        logger.info(f"┃       epoch  ┃        loss  │    top_1     ╷    top_5     ┃        l.r.  │     elapsed  ┃        loss  │     top_1    │     top_5    ┃")
        logger.info(f"┠──────────────╂──────────────┼──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┼──────────────┨")