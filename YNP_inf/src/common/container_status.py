import socket
import subprocess
import requests
from urllib.parse import urljoin

class ContainerStatus():

	def __init__(self, host_web):
		self.host_web = host_web
		self.short_id = None
		self.full_id = None
		self.get_id()
		self.events = ['/events/{}/before_start',		#Вызывается как можно раньше после запуска контейнера и перед началом работы скриптов обработки данных
				'/events/{}/on_progress',		#Вызывается периодически по мере обработки данных в контейнере
				'/events/{}/on_error',	#Вызывается в случае возникновения ошибки в процессе обработки данных в контейнере
				'/events/{}/before_end']	#Вызывается после окончания обработки данных в контейнере и сохранением выходных файлов в папке OUT_DIR и перед остановкой (удалением) контейнера

	def get_short_id(self):
		'''Получение краткого id контейнера, как имя хоста
		'''
		self.short_id = socket.gethostname()

	def get_containers_from_docker(self)->dict:
		'''Получение списка активных контейнеров от docker
		Return: словарь вида full_id:name
		'''
		result = subprocess.check_output('docker ps --format "{{.ID}}: {{.Names}}" --no-trunc', shell=True).decode("utf-8").split('\n')
		containers = {}
		for line in result:
			if ': ' in line:
				key, value = line.split(': ')
				containers[key] = value
		return containers

	def get_id(self, full:bool = True):
		'''Получение полного id контейнера
		'''
		self.get_short_id()
		if full:
			containers = self.get_containers_from_docker()
			for id_ in containers.keys():
				if self.short_id == id_[:12]:
					self.full_id = id_
					break
		
	def post(self, url:str, data:dict = {}):
		'''Шаблон post запроса
		Return: ответ на запрос
		'''
		#print(url)
		res = {}
		try:
			res = requests.post(url, json = data)
		except Exception as e:
			print("Error: " +str(e))
		finally:
			return res
		
	def post_status(self, status:int, data:dict = None):
		'''Шаблон post запроса со статусом
		Args: 	status - № статуча из events
				data - сообщение
		Return: ответ на запрос
		'''
		return self.post(urljoin(self.host_web,self.events[status].format(self.full_id)),data)
		
	def post_start(self,data:dict = None):
		'''Шаблон post запроса со статусом before_start
		Args: 	data - сообщение пустое
		Return: ответ на запрос
		'''
		return self.post_status(0, data)
		
	def post_progress(self,data = None):
		'''Шаблон post запроса со статусом on_progress
		Args: 	data - сообщение формата {'on_progress':'0.xx'} - процент выполнения в виде десятичной дроби
		Return: ответ на запрос
		'''
		return self.post_status(1,data)
		
	def post_error(self,data = None):
		'''Шаблон post запроса со статусом on_error
		Args: 	data - сообщение формата {'on_error':'error text'} - текст ошибки
		Return: ответ на запрос
		'''
		return self.post_status(2,data)
		
	def post_end(self,data = None):
		'''Шаблон post запроса со статусом before_end
		Args: 	data - сообщение формата {'out_files':['xxx.json', 'yyy.json']} - имена выходных файлов
		Return: ответ на запрос
		'''
		return self.post_status(3,data)

def get_id():
	'''Получение полного id контейнера
	'''
	full_id = ""
	containers = {}
	
	short_id = socket.gethostname()
	result = subprocess.check_output('docker ps --format "{{.ID}}: {{.Names}}" --no-trunc', shell=True).decode("utf-8").split('\n')
	for line in result:
		if ': ' in line:
			key, value = line.split(': ')
			containers[key] = value
	for id_ in containers.keys():
		if short_id == id_[:12]:
			full_id = id_
			break
	return full_id
